# the matmulinteger pipeline

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

def insert_cast_after_matmul(model, output_path):
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
                cast_output_name = output_name + '_casted'
                cast_output_tensor = helper.make_tensor_value_info(cast_output_name, TensorProto.FLOAT, get_activation_tensor_shape_by_name(graph, output_name))

                # Create the Cast node for input
                cast_node = helper.make_node(
                    'Cast',
                    inputs=[output_name],
                    outputs=[cast_output_name],
                    to=onnx.TensorProto.FLOAT,
                    name=output_name + '_cast'
                )

                # Insert DequantizeLinear node after MatMul node
                new_nodes.append(cast_node)
                node_list = find_nodes_by_input_name(graph, output_name) # find nodes that have proj's output as input
                for n, ind in node_list:
                    n.input[ind] = cast_output_name

    # Replace the graph nodes with the new nodes list
    graph.ClearField('node')
    graph.node.extend(new_nodes)

    return model
    # onnx.checker.check_model(output_path, full_check=True)

def replace_matmul_with_matmulinteger(model, output_path):
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

                # Create MatMulInteger node
                matmulinteger_node = helper.make_node(
                    'MatMulInteger',
                    inputs=matmul_node.input,
                    outputs=matmul_node.output,
                    name=(matmul_node.name).replace("MatMul", "MatMulInteger")
                )

                # Replace MatMul node with MatMulInteger node
                insert_node(graph, matmul_node.name, matmulinteger_node)
                graph.node.remove(matmul_node)

    # for i, node in enumerate(graph.node):
    #     print(f"[{i}] {node.name}")

    return model

def initialize_random_int8_weights_for_matmulinteger(model, output_path):
    # model = onnx.load(model_path)
    graph = model.graph

    # Traverse the graph to find MatMul nodes
    for node in graph.node:
        if len(((node.name).split('/'))) >= 2 \
        and len(((node.name).split('/')[-2]).split('_')) == 2 \
        and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
        and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
        and node.op_type == 'MatMulInteger': 

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'MatMulInteger_2'): # for down had transform

                rhs_input_name = node.input[1] # rhs is located at the fourth input for MatMulInteger
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

def load_proj_from_checkpoint(proj_name, layer_num):
    # NOTE: redirect this path to the model's path
    checkpoint_files = ['/local/mnt2/workspace2/jouachen/prefixquant0218_ws/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00001-of-00002.safetensors', '/local/mnt2/workspace2/jouachen/prefixquant0218_ws/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00002-of-00002.safetensors']

    full_checkpoint = {}
    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=None)
        full_checkpoint.update(checkpoint)

    # (1) *******************quantized int8 weight********************
    
    group_size = {
        'q_proj': 4096,
        'k_proj': 4096,
        'v_proj': 4096,
        'down_proj': 14336,
        'o_proj': 4096,
        'up_proj': 4096,
        'gate_proj': 4096}
    
    weight = None
    scale = None
    weight_scale = None

    for k in full_checkpoint.keys(): # get either the scale or the weight
        if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(proj_name + ".weight_quantizer.scale"): # match layer num and scale postfix

            fp16_tensor = full_checkpoint[k.replace("weight_quantizer.scale", "weight")]
            scale = full_checkpoint[k]

            # print(fp16_tensor, scale)
            weight_scale = scale.transpose(1, 0).numpy()

            weight = static_fake_quant(fp16_tensor, scale, group_size[proj_name]).to(torch.int8)
            weight = weight.detach().numpy()

            dim1, dim2 = full_checkpoint[k.replace("weight_quantizer.scale", "weight")].shape
            weight = weight.transpose(1, 0) # <-- because F.linear (X, W) = matmul (X, W^T) ## to confirm # weight.reshape(dim2, dim1) # 

    # (2) *******************load proj scale*********************
    scale_postfix = {
        'q_proj': 'input_layernorm.output_quantizer.scale',
        'k_proj': 'input_layernorm.output_quantizer.scale',
        'v_proj': 'input_layernorm.output_quantizer.scale',
        'down_proj': 'mlp.down_proj.input_quantizer.scale',
        'o_proj': 'self_attn.o_proj.input_quantizer.scale',
        'up_proj': 'post_attention_layernorm.output_quantizer.scale',
        'gate_proj': 'post_attention_layernorm.output_quantizer.scale'}

    for k in full_checkpoint.keys():
        if (len(k.split('.')) >=3 and k.split('.')[2] == layer_num) and k.endswith(scale_postfix[proj_name]): # match layer num and scale postfix
            # print(proj_name, full_checkpoint[k].squeeze().float().numpy())
            scale = full_checkpoint[k].squeeze().half().numpy().item()
            # print(f"{full_checkpoint[k]}")
    
    return weight, scale, weight_scale

def load_initializer_from_checkpoint(model, output_path):
    total_layer = 32
    graph = model.graph

    # NOTE: redirect this path to the model's path
    checkpoint_files = ['/local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00001-of-00002.safetensors', '/local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8/model-00002-of-00002.safetensors']
    full_checkpoint = {}
    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=None)
        full_checkpoint.update(checkpoint)

    # construct quantized_weight_map
    quantized_weight_map = [{} for _ in range(total_layer)]
    for node in graph.node:
        if len(((node.name).split('/'))) >= 2 \
            and len(((node.name).split('/')[-2]).split('_')) == 2 \
            and ((node.name).split('/')[-2]).split('_')[1] == 'proj' \
            and ((node.name).split('/')[-2]).split('_')[0] not in exclude_quant_proj_layer \
            and node.op_type == 'MatMulInteger':

            if  ((node.name).split('/')[-2]).split('_')[0] != 'down' or \
                (((node.name).split('/')[-2]).split('_')[0] == 'down' and ((node.name).split('/')[-1]) == 'MatMulInteger_2'): # for down had transform
                weight_name = node.input[1] # rhs is located at the fourth input for MatMulInteger
                proj_name = (node.name).split('/')[-2]
                layer_num = int((node.name).split('/')[-4].split('.')[-1])
                quantized_weight_map[layer_num][proj_name] = weight_name
    # print(quantized_weight_map)

    # construct inputscale_map
    inputscale_map = [{} for _ in range(total_layer)]
    for initializer in graph.initializer:
        if len((initializer.name).split('/')) >= 5:
            namelist = (initializer.name).split('/')
            scale_name = namelist[-1].split('_')[-1]
            proj_name = namelist[-2]
            layer_num = int(namelist[-4].split('.')[1])
            if scale_name == "scaled":
                inputscale_map[layer_num][proj_name] = initializer.name
    # print(inputscale_map)

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
                outputscale_map[layer_num][proj_name] = weight_name
    # print(outputscale_map)

    # construct initializer_map
    initializer_map = {}
    for i, initializer in enumerate(graph.initializer):
        initializer_map[initializer.name] = i
    # print(initializer_map)
    # breakpoint()

    # for weight quantization
    group_size = {
        'q_proj': 4096,
        'k_proj': 4096,
        'v_proj': 4096,
        'down_proj': 14336,
        'o_proj': 4096,
        'up_proj': 4096,
        'gate_proj': 4096}

    proj_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    num_hidden_layers = sum([1 if len(lmap) != 0 else 0 for lmap in quantized_weight_map])
    for layer_num_int in range(num_hidden_layers): # 1-32
        for proj_name in proj_layers: # 7
            weight, in_scale, weight_scale = load_proj_from_checkpoint(proj_name, str(layer_num_int))
            
            # set weight
            initializer_name = quantized_weight_map[layer_num_int][proj_name]
            updated_initializer = numpy_helper.from_array(weight, initializer_name)
            graph.initializer[initializer_map[initializer_name]].CopyFrom(updated_initializer)
            print(f"Set \"{initializer_name}\" to value of \"{weight}\" quantized on scale \"{weight_scale}\" from checkpoint")

            # set input scale
            initializer_name = inputscale_map[layer_num_int][proj_name]
            updated_initializer = numpy_helper.from_array(np.array([in_scale], dtype=np.float32), initializer_name)
            graph.initializer[initializer_map[initializer_name]].CopyFrom(updated_initializer)
            print(f"Set \"{initializer_name}\" to value of \"{in_scale}\" from checkpoint")

            # set dequant scale
            initializer_name = outputscale_map[layer_num_int][proj_name]
            weight_scale = np.array(weight_scale.reshape(1, 1, weight_scale.shape[1]), dtype=np.float16).astype(np.float32)
            in_scale = np.array(in_scale, dtype=np.float16).astype(np.float32)
            dequant_scale = weight_scale * in_scale
            updated_initializer = numpy_helper.from_array(dequant_scale, initializer_name)
            graph.initializer[initializer_map[initializer_name]].CopyFrom(updated_initializer)
            print(f"Set \"{initializer_name}\" to value of scale \"{dequant_scale}\" using in_scale * weight_scale")

    return model

def quantize_model_to_int8(source_onnx):

    m = insert_quantizelinear_before_matmul(source_onnx, source_onnx)
    m = insert_cast_after_matmul(m, source_onnx)
    m = replace_matmul_with_matmulinteger(m, source_onnx)
    m = initialize_random_int8_weights_for_matmulinteger(m, source_onnx)
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