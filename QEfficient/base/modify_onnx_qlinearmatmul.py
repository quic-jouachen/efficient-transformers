# adding qlinearmatmul and dequantize scale

import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from accelerate.utils.modeling import load_state_dict
import torch
import onnxruntime


exclude_quant_proj_layer = [] # 'q', 'k', 'up', 'o', 'gate', 'down' # 'v', 'k'

CLIPMIN = 1e-4

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def static_fake_quant(x, scale, group_size):
    '''
    static quantization
    '''
    qmin = -128
    qmax = 127
    zero_point = None

    scale = clamp_ste(scale,1e-4, 1e4)
    round_zero_point = clamp_ste(round_ste(zero_point), qmin, qmax) if zero_point is not None else None
    # if self.quant_type == 'weight':
    dim1, dim2 = x.shape
    x_reshaped = x.reshape(-1, group_size)
    # elif self.quant_type == 'activation':
    #     bs, n, dim1 = x.shape
    #     x_reshaped = x.reshape(bs, n, -1, group_size)


    x_int = round_ste(x_reshaped / scale)
    if round_zero_point is not None:
        x_int = x_int.add(round_zero_point)
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = x_int
    if round_zero_point is not None:
        x_dequant = x_dequant.sub(round_zero_point)
    # print("after quant: ", x_dequant)
    # breakpoint()
    # x_dequant = x_dequant.mul(scale) #### <--- remove this here
    if group_size:
        # if self.quant_type == 'weight':
        x_dequant = x_dequant.reshape(dim1, dim2)
        # elif self.quant_type == 'activation':
        #     x_dequant = x_dequant.reshape(bs, n, dim1)
    return x_dequant

def insert_node(graph, target_node_name, node_to_insert):
    insertion_index = None
    for i, node in enumerate(graph.node):
        if node.name == target_node_name:
            insertion_index = i # -1 #+ 1  # Insert before the target node
            break

    if insertion_index is None:
        print("Target node not found")
    else: # Insert the node into the graph at the determined index
        print(f"Insert node {node_to_insert.name} before {target_node_name}")
        graph.node.insert(insertion_index, node_to_insert)

def find_nodes_by_input_name(graph, input_name):
    matching_nodes = []
    for node in graph.node:
        for ind in range(len(node.input)):
            if input_name == node.input[ind]:
                matching_nodes.append((node, ind))

    return matching_nodes

def get_activation_tensor_shape_by_name(graph, tensor_name):
    for input_tensor in graph.value_info:
        if input_tensor.name == tensor_name:
            tensor_type = input_tensor.type.tensor_type
            shape = [dim.dim_value for dim in tensor_type.shape.dim]
            return shape

def insert_quantizelinear_before_matmul(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    insterted_target_node = []
    # Traverse the graph to find MatMul nodes
    # new_nodes = []
    for node in graph.node:
        # new_nodes.append(node)
        if len(((node.name).split('/'))) >= 2 \
            and len(((node.name).split('/')[-2]).split('_')) == 2 \
            and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
            and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
            and node.op_type == 'MatMul' \
            and node.name not in insterted_target_node:

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'MatMul_2'): # for down had transform

                ###  add only one QuantizeLinear to LHS ###
                input_name = node.input[0]
                output_name = node.name + '_quantized'

                # Create QuantizeLinear node
                scale = helper.make_tensor(node.name + '_scaled', TensorProto.FLOAT, [1], [1])
                zero_point = helper.make_tensor(node.name + '_zero_point', TensorProto.INT8, [1], [0])
                quantized_output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT8, get_activation_tensor_shape_by_name(graph, input_name))
                quantize_node = helper.make_node(
                    'QuantizeLinear',
                    inputs=[input_name, scale.name, zero_point.name],
                    outputs=[output_name],
                    name=node.name + '_QuantizeLinear'
                )

                # Insert QuantizeLinear node before MatMul node
                insert_node(graph, node.name, quantize_node)
                node.input[0] = output_name
                insterted_target_node.append(node.name)

                # Add scale and zero_point initializers
                graph.initializer.extend([scale, zero_point])

    return model
    # onnx.checker.check_model(output_path, full_check=True)

def insert_dequantizelinear_after_matmul(model, output_path):
    # model = onnx.load(model_path)
    graph = model.graph

    # Traverse the graph to find MatMul nodes
    new_nodes = []
    for node in graph.node:
        new_nodes.append(node)
        if len(((node.name).split('/'))) >= 2 \
        and len(((node.name).split('/')[-2]).split('_')) == 2 \
        and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
        and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
        and node.op_type == 'MatMul' \
        and (node.name).split('/')[-1] != 'MatMul_QuantizeLinear' :

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'MatMul_2'): # for down had transform
            
                output_name = node.output[0]
                dequantized_output_name = output_name + '_dequantized'

                # Create DequantizeLinear node
                scale = helper.make_tensor(output_name + '_scale', TensorProto.FLOAT, [1], [1])
                zero_point = helper.make_tensor(output_name + '_zero_point', TensorProto.INT8, [1], [0])
                dequantized_output_tensor = helper.make_tensor_value_info(dequantized_output_name, TensorProto.FLOAT, get_activation_tensor_shape_by_name(graph, output_name))
                dequantize_node = helper.make_node(
                    'DequantizeLinear',
                    inputs=[output_name, scale.name, zero_point.name],
                    outputs=[dequantized_output_name],
                    name=output_name + '_DequantizeLinear'
                )

                # Insert DequantizeLinear node after MatMul node
                new_nodes.append(dequantize_node)
                node_list = find_nodes_by_input_name(graph, output_name) # find nodes that have proj's output as input
                for n, ind in node_list:
                    n.input[ind] = dequantized_output_name

                # Add scale and zero_point initializers
                graph.initializer.extend([scale, zero_point])

    # Replace the graph nodes with the new nodes list
    graph.ClearField('node')
    graph.node.extend(new_nodes)

    return model
    # onnx.checker.check_model(output_path, full_check=True)

def replace_matmul_with_qlinearmatmul(model, output_path):
    graph = model.graph

    # for i, node in enumerate(graph.node):
    #     print(f"[{i}] {node.name}")
    
    # print("***************************************")

    insterted_target_node = []
    # Traverse the graph to find MatMul nodes
    # new_nodes = []
    for node in graph.node:
        # new_nodes.append(node)
        if len(((node.name).split('/'))) >= 2 \
            and len(((node.name).split('/')[-2]).split('_')) == 2 \
            and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
            and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
            and node.op_type == 'MatMul' \
            and node.name not in insterted_target_node:

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'MatMul_2'): # for down had transform

                matmul_node = node

                # Define quantization parameters (example values)
                a_scale = helper.make_tensor(matmul_node.name + '_inscale', TensorProto.FLOAT, [1], [1])
                a_zero_point = helper.make_tensor(matmul_node.name + '_a_zero_point', TensorProto.INT8, [1], [0])
                b_scale = helper.make_tensor(matmul_node.name + '_b_scale', TensorProto.FLOAT, [1], [1])
                b_zero_point = helper.make_tensor(matmul_node.name + '_b_zero_point', TensorProto.INT8, [1], [0])
                y_scale = helper.make_tensor(matmul_node.name + '_outscale', TensorProto.FLOAT, [1], [1])
                y_zero_point = helper.make_tensor(matmul_node.name + '_y_zero_point', TensorProto.INT8, [1], [0])

                # Create QLinearMatMul node
                qlinear_matmul_node = helper.make_node(
                    'QLinearMatMul',
                    inputs=[matmul_node.input[0], matmul_node.name + '_inscale', matmul_node.name + '_a_zero_point', matmul_node.input[1], matmul_node.name + '_b_scale', matmul_node.name + '_b_zero_point', matmul_node.name + '_outscale', matmul_node.name + '_y_zero_point'],
                    outputs=matmul_node.output,
                    name=(matmul_node.name).replace("MatMul", "QLinearMatMul")
                )

                # Replace MatMul node with QLinearMatMul node
                insert_node(graph, matmul_node.name, qlinear_matmul_node)
                # graph.node.append(qlinear_matmul_node)
                graph.node.remove(matmul_node)

                # Add quantization parameters to the graph
                graph.initializer.extend([a_scale, a_zero_point, b_scale, b_zero_point, y_scale, y_zero_point])

    # for i, node in enumerate(graph.node):
    #     print(f"[{i}] {node.name}")

    return model

def initialize_random_int8_weights_for_matmul(model, output_path):
    # model = onnx.load(model_path)
    graph = model.graph

    # Traverse the graph to find MatMul nodes
    for node in graph.node:
        if len(((node.name).split('/'))) >= 2 \
        and len(((node.name).split('/')[-2]).split('_')) == 2 \
        and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
        and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
        and node.op_type == 'QLinearMatMul': 

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'QLinearMatMul_2'): # for down had transform

                rhs_input_name = node.input[3] # rhs is located at the fourth input for QLinearMatMul
                weight_shape = None

                # Find the initializer corresponding to the RHS input
                for initializer in graph.initializer:
                    if initializer.name == rhs_input_name:
                        weight_shape = numpy_helper.to_array(initializer).shape
                        graph.initializer.remove(initializer)
                        break

                if weight_shape is not None:
                    # Create random int8 weights
                    random_weights = np.random.randint(-128, 127, size=weight_shape, dtype=np.int8)
                    new_initializer = numpy_helper.from_array(random_weights, name=rhs_input_name)

                    # Add the new initializer to the graph
                    graph.initializer.append(new_initializer)

    return model
    # onnx.checker.check_model(output_path, full_check=True)

def load_initializer_from_checkpoint(model, output_path):
    total_layer = 32
    graph = model.graph

    # NOTE: redirect this path to the model's path
    checkpoint_files = ['/local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00001-of-00002.safetensors', '/local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00002-of-00002.safetensors']

    full_checkpoint = {}
    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=None)
        full_checkpoint.update(checkpoint)

    # (1) *******************quantized int8 weight********************
    # quantized_weight_map = {
    #     'onnx::MatMul_543':'q_proj',
    #     'onnx::MatMul_544':'k_proj',
    #     'onnx::MatMul_545':'v_proj',
    #     'onnx::MatMul_583':'o_proj',
    #     'onnx::MatMul_584':'gate_proj',
    #     'onnx::MatMul_585':'up_proj',
    #     'onnx::MatMul_593':'down_proj'} # replace it with operator-based mapping later

    # construct quantized_weight_map
    quantized_weight_map = [{} for _ in range(total_layer)]
    for node in graph.node:
        if len(((node.name).split('/'))) >= 2 \
            and len(((node.name).split('/')[-2]).split('_')) == 2 \
            and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
            and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
            and node.op_type == 'QLinearMatMul':

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'QLinearMatMul_2'): # for down had transform
                weight_name = node.input[3] # rhs is located at the fourth input for QLinearMatMul
                proj_name = (node.name).split('/')[-2]
                layer_num = int((node.name).split('/')[-4].split('.')[-1])
                quantized_weight_map[layer_num][weight_name] = proj_name
    # print(quantized_weight_map)
        
    group_size = {
        'q_proj': 4096,
        'k_proj': 4096,
        'v_proj': 4096,
        'down_proj': 14336,
        'o_proj': 4096,
        'up_proj': 4096,
        'gate_proj': 4096}

    for layer_num_int in range(total_layer):
        for initializer in graph.initializer:
            if initializer.name in quantized_weight_map[layer_num_int]:
                proj_name = quantized_weight_map[layer_num_int][initializer.name]
                layer_num = str(layer_num_int) # adjust here for layer

                for k in full_checkpoint.keys(): # get either the scale or the weight
                    if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(proj_name + ".weight_quantizer.scale"): # match layer num and scale postfix

                        fp16_tensor = full_checkpoint[k.replace("weight_quantizer.scale", "weight")]
                        scale = full_checkpoint[k]

                        weight = static_fake_quant(fp16_tensor, scale, group_size[proj_name]).to(torch.int8)
                        int8_tensor = weight.detach().numpy()

                        # verify shape
                        dim1, dim2 = full_checkpoint[k.replace("weight_quantizer.scale", "weight")].shape
                        int8_tensor = int8_tensor.transpose(1, 0) #.reshape(dim2, dim1) ## to confirm
                        
                        # print(proj_name, int8_tensor)
                        assert numpy_helper.to_array(initializer).shape == int8_tensor.shape, "weight shape mismatch!"

                        # update
                        updated_initializer = numpy_helper.from_array(int8_tensor, initializer.name)
                        initializer.CopyFrom(updated_initializer)
                        print(f"Set \"{initializer.name}\" to value of \"{k.replace('weight_quantizer.scale', 'weight')}\" quantized on scale \"{k}\" from checkpoint")
                        print(proj_name, int8_tensor)

    # (2) *******************load proj input scale*********************
    scale_postfix = {
        'q_proj': 'input_layernorm.output_quantizer.scale',
        'k_proj': 'input_layernorm.output_quantizer.scale',
        'v_proj': 'input_layernorm.output_quantizer.scale',
        'down_proj': 'mlp.down_proj.input_quantizer.scale',
        'o_proj': 'self_attn.o_proj.input_quantizer.scale',
        'up_proj': 'post_attention_layernorm.output_quantizer.scale',
        'gate_proj': 'post_attention_layernorm.output_quantizer.scale'}

    for initializer in graph.initializer:
        if len((initializer.name).split('/')) >= 5:
            namelist = (initializer.name).split('/')
            scale_name = namelist[-1].split('_')[-1]
            proj_name = namelist[-2]
            layer_num = namelist[-4].split('.')[1]

            if scale_name == "inscale" and proj_name in scale_postfix: # only update input scale
                for k in full_checkpoint.keys():
                    if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(scale_postfix[proj_name]): # match layer num and scale postfix
                        updated_initializer = numpy_helper.from_array(full_checkpoint[k].squeeze().float().numpy(), initializer.name)
                        initializer.CopyFrom(updated_initializer)
                        print(f"Set \"{initializer.name}\" to value of \"{k}\" from checkpoint")
                        print(proj_name, full_checkpoint[k].squeeze().float().numpy())
    
    # # (3) *******************load proj output scale (same as input scale for now) ********************* ==> don't need this, AIC already handles
    # scale_postfix = {
    #     'q_proj': 'input_layernorm.output_quantizer.scale',
    #     'k_proj': 'input_layernorm.output_quantizer.scale',
    #     'v_proj': 'input_layernorm.output_quantizer.scale',
    #     # 'k_proj': 'self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.k_quantizer.scale', # out scale given (supposedly for kv cache)
    #     # 'v_proj': 'self_attn.v_proj.output_quantizer.scale', # out scale given (supposedly for kv cache)
    #     'down_proj': 'mlp.down_proj.input_quantizer.scale',
    #     'o_proj': 'self_attn.o_proj.input_quantizer.scale',
    #     'up_proj': 'post_attention_layernorm.output_quantizer.scale',
    #     'gate_proj': 'post_attention_layernorm.output_quantizer.scale'}

    # for initializer in graph.initializer:
    #     if len((initializer.name).split('/')) >= 5:
    #         namelist = (initializer.name).split('/')
    #         scale_name = namelist[-1].split('_')[-1]
    #         proj_name = namelist[-2]
    #         layer_num = namelist[-4].split('.')[1]

    #         if scale_name == "outscale" and proj_name in scale_postfix: # only update input scale
    #             for k in full_checkpoint.keys():
    #                 if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(scale_postfix[proj_name]): # match layer num and scale postfix
    #                     updated_initializer = numpy_helper.from_array(full_checkpoint[k].squeeze().float().numpy(), initializer.name)
    #                     initializer.CopyFrom(updated_initializer)
    #                     print(f"Set \"{initializer.name}\" to value of \"{k}\" from checkpoint")
    #                     print(proj_name, full_checkpoint[k].squeeze().float().numpy())
    
    # (3) *******************load outputscale multiplier as weight scale (assume grouping on hidden dim) *********************
    # construct outputscale_map
    outputscale_map = [{} for _ in range(total_layer)]
    for node in graph.node:
        if len(((node.name).split('/'))) >= 2 \
            and len(((node.name).split('/')[-2]).split('_')) == 2 \
            and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
            and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
            and node.op_type == 'Mul':

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'Mul_1'): # for down had transform
                weight_name = node.input[1]
                proj_name = (node.name).split('/')[-2]
                layer_num = int((node.name).split('/')[-4].split('.')[-1])
                outputscale_map[layer_num][weight_name] = proj_name
    # print(outputscale_map)
    
    for layer_num_int in range(total_layer):
        for initializer in graph.initializer:
            if initializer.name in outputscale_map[layer_num_int]:
                proj_name = outputscale_map[layer_num_int][initializer.name]
                layer_num = str(layer_num_int) # adjust here for layer

                for k in full_checkpoint.keys(): # get either the scale or the weight
                    if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(proj_name + ".weight_quantizer.scale"): # match layer num and scale postfix

                        scale = full_checkpoint[k]
                        scale = scale.view(1, 1, scale.shape[0])
                        scale = scale.detach().numpy()

                        # update
                        updated_initializer = numpy_helper.from_array(scale, initializer.name)
                        initializer.CopyFrom(updated_initializer)
                        print(f"Set \"{initializer.name}\" to value of scale \"{k}\" from checkpoint")
                        print(scale)
    
    return model

def quantize_model_to_int8(source_onnx):

    m = insert_quantizelinear_before_matmul(source_onnx, source_onnx)
    m = insert_dequantizelinear_after_matmul(m, source_onnx)
    m = replace_matmul_with_qlinearmatmul(m, source_onnx)
    m = initialize_random_int8_weights_for_matmul(m, source_onnx)
    m = load_initializer_from_checkpoint(m, source_onnx)

    output_path = source_onnx
    model = m
    # Delete the original data file
    import os
    for i in range(10):
        original_data_file_path = f"{output_path.split('.')[0]}_{i}.onnx.data"
        if os.path.exists(original_data_file_path):
            os.remove(original_data_file_path)
            print(f"Original data file '{original_data_file_path}' has been deleted.")
        else:
            break
            # print(f"Original data file '{original_data_file_path}' not found.")

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{(output_path.split('/')[-1]).split('.')[0]}_0.onnx.data",
        size_threshold=1024,
        convert_attribute=False,
    )

    # onnx.checker.check_model(output_path, full_check=True)