# -------------------------------------------------------------------
# python prefixquant_model.py --quant_model /local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8
# -------------------------------------------------------------------

import sys

from transformers import AutoModelForCausalLM
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import load_hf_tokenizer
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download


import os
import sys
import random
import numpy as np
import torch
# import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
# from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, init_k_quantizer, init_v_quantizer # register_online_had,
# import utils.model_utils as model_utils
# import utils.rotation_utils as rotation_utils
# from main import evaluate
# from utils.train_utils import load_json_as_namespace,create_logger
import json
from types import SimpleNamespace
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
torch.backends.cudnn.benchmark = True


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
parser.add_argument("--real_quant", default=False, action="store_true",
                    help="use real quantization instead of fake quantization, can reduce memory footprint")
parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
# parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
# parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# # init logger
# if args.output_dir:
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
# output_dir = Path(args.output_dir)
# logger = create_logger(output_dir)

with open(os.path.join(args.quant_model_path, 'prefixequant_config.json'), 'r') as json_file:
    data = json.load(json_file)
    quant_config = json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))



# if quant_config.set_prefixed_tokens:
#     prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
# else:
#     prefixed_key_values = None

model_path = "/local/mnt2/workspace2/jouachen/sq0121_ws/prefixquant-qeff/llama-3p1-8b-instruct-fp16-prefix-w8a8kv8"
tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False,legacy=False,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.quant_model_path,
                                            attn_implementation="eager", 
                                            num_hidden_layers=1, # dump 1 layer to check, uncomment to generate full model onnx
                                            device_map='cpu',
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True)


# *********************** Add output weight scale node ********************** #
from optimized_hadamard_transform import hadamard_transform_down_proj, hadamard_transform_down_proj_new
import torch.nn as nn
import torch.nn.functional as F
# import utils.hadamard_utils as hadamard_utils

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        quantized_shape,
        asym=True,
        group_size=-1,
        quantized_item_stat=None,
        quant_type='weight',
        mode='static',
        minmax_init=True,
        disable_zero_point_in_sym=True,
        activation_clipping=False,
    ):
        '''
        quantized_item_stat: 
        - weight: original weight
        - activation: channel-wise maximum values
        '''
        super().__init__()
        assert 2 <= n_bits <= 32, "bitwidth not supported"
        self.n_bits = n_bits
        self.quantized_shape = quantized_shape
        self.group_size = group_size if group_size != -1 else quantized_shape[-1]
        assert quantized_shape[-1] % group_size == 0
        self.inc_groups = quantized_shape[-1] // self.group_size
        self.quant_type = quant_type
        self.mode = mode
        self.asym = asym
        self.disable_zero_point_in_sym = disable_zero_point_in_sym
        self.activation_clipping = activation_clipping
        self.enable = True
        if self.asym or not self.disable_zero_point_in_sym:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        else:
            self.qmin = -(2 ** (self.n_bits - 1))
            self.qmax = 2 ** (self.n_bits - 1) - 1
            
        # init scale and zero point through Max-Min quantization
        if quant_type == 'weight':
            self.find_weight_quant_param(quantized_item_stat,minmax_init)
    
    @torch.no_grad()     
    def find_weight_quant_param(self, quantized_item_stat,minmax_init):
        assert len(self.quantized_shape) == 2, 'only support for linear layer'
        self.mode = 'static'
        if minmax_init:
            assert quantized_item_stat is not None
            x = quantized_item_stat.reshape(-1,self.group_size)
            if self.asym:
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                self.original_max = xmax  # not used
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            else:
                xmax =  x.abs().amax([-1], keepdim=True)
                self.original_max = xmax
                scale = 2*xmax/(2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                self.scale = nn.Parameter(scale)
                if self.disable_zero_point_in_sym:
                    self.zero_point = None
                    self.qmin = -(2 ** (self.n_bits - 1))
                    self.qmax = 2 ** (self.n_bits - 1) - 1
                else:
                    self.register_buffer("zero_point", (2**(self.n_bits-1)-1)*torch.ones_like(self.scale))    
        else:
            dims = self.quantized_shape[0] * self.quantized_shape[1] // self.group_size
            self.scale = nn.Parameter(torch.ones(dims,1))
            if self.asym or not self.disable_zero_point_in_sym:
                self.zero_point = nn.Parameter(torch.zeros(dims,1))
            else:
                self.zero_point = None
        
        self.scale = nn.Parameter(torch.randn(self.scale.shape)) ## to get rid of identity node
    def forward(self, x: torch.Tensor):
        return self.scale, self.group_size

class QLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False
        
    
    def forward(self, input: torch.Tensor):

        scale, group_size = self.weight_quantizer(self.weight)
        
        # print(scale.shape, group_size)
        scale = scale.view(1, 1, scale.shape[0])

        out = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
        # out = out.reshape(-1, self.group_size)

        # bs, n, dim1 = out.shape
        # out = out.reshape(bs, n, -1, self.group_size)
        # # breakpoint()
        out =  out * scale
        # out = out.reshape(bs, n, dim1)
        # print(out.shape)
        # print(self.scale.shape)

        return out


def wrap_to_quant_model(model):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    for name, module in model.named_modules():
        # skip lm_head quantization
        if 'lm_head' in name:
            continue
        # skip quantization of norm for lm_head
        if 'model.norm' in name:
            continue
        if 'model.input.input_layernorm' in name:
            continue
        if 'model.input.post_attention_layernorm' in name:
            continue
        if isinstance(module,torch.nn.Linear): #and 'k_proj' not in name and 'v_proj' not in name: # excluding those having output scale
            quantlinear = QLinear(module)
            set_op_by_name(model, name, quantlinear)
            del module  

wrap_to_quant_model(model)
# *********************** Add output weight scale node ********************** #

# *********************** register hardamard for down_proj ********************** #

class HadLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False

    
    
    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype
        # Hadamrd in fp16
        input = hadamard_transform_down_proj_new(input, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias

        scale, group_size = self.weight_quantizer(self.weight)
        scale = scale.view(1, 1, scale.shape[0])

        out = F.linear(input, self.weight, self.bias, **self.fwd_kwargs)

        out = out * scale

        return out

def add_hadamard_to_down_proj():
    for name, module in model.named_modules():
        # skip lm_head quantization
        if 'lm_head' in name:
            continue
        # skip quantization of norm for lm_head
        if 'model.norm' in name:
            continue
        if isinstance(module,QLinear) and 'down_proj' in name:
            hadlinear = HadLinear(module)
            set_op_by_name(model, name, hadlinear)  
            del module  

def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module,HadLinear) and 'down_proj' in name:
            # print(model.config.intermediate_size)
            # had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size) # 14336


            had_K = torch.tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                    -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,
                    1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.],
                    [ 1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,
                    1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.],
                    [ 1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,
                    1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.],
                    [ 1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,
                    1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.],
                    [ 1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,
                    1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.],
                    [ 1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1., -1.,
                    1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1.],
                    [ 1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1.,
                    1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1.],
                    [ 1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1.,
                    1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1.],
                    [ 1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
                    1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1.],
                    [ 1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,
                    1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.],
                    [ 1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,
                    1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.],
                    [ 1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,  1.,
                    1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1.],
                    [ 1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,  1.,
                    1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.],
                    [-1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                    -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [ 1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1.,
                    -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1.],
                    [ 1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,
                    -1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1.],
                    [ 1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,
                    -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.],
                    [ 1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,
                    -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1.],
                    [ 1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,
                    -1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1.],
                    [ 1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1., -1.,
                    -1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1.],
                    [ 1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1., -1.,
                    -1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.],
                    [ 1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1., -1.,
                    -1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.],
                    [ 1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1.,
                    -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.],
                    [ 1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,
                    -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1.],
                    [ 1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1.,
                    -1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1.],
                    [ 1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,  1.,
                    -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.],
                    [ 1.,  1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,
                    -1., -1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1.]], device='cpu', dtype=torch.float32)
            K = 28

            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False

# #### only dealing with down_proj's hardamard here ###
add_hadamard_to_down_proj()
if quant_config.down_online_had:
    register_online_had(model)
# #### only dealing with down_proj's hardamard here ###

# init weight quantizer
def init_weight_quantizer(args, model, minmax_init=True):
    for name, module in model.named_modules():
        if isinstance(module,QLinear) or isinstance(module,HadLinear):
            wbits = args.wbits
            module.wbits = wbits
            if wbits >= 16:
                continue
            w_group_size=args.w_group_size
            w_asym=args.w_asym
            quantized_item_stat=module.weight if minmax_init else None
            module.use_weight_quant = True
            module.weight_quantizer = UniformAffineQuantizer(wbits, module.weight.shape,  w_asym, w_group_size,
                                                            quantized_item_stat=quantized_item_stat,
                                                            quant_type='weight',
                                                            minmax_init=minmax_init)
            sym_stat = "asymmetric" if w_asym else 'symmetric'
            print(f'weight quantization: set {name} as w{wbits}g{w_group_size} {sym_stat} quantization')

if quant_config.wbits < 16:
    print('init weight quantizer')
    init_weight_quantizer(quant_config, model, minmax_init=False)
# *********************** register hardamard for down_proj ********************** #

seq_len = 128
ctx_len = 256
full_batch_size = 4
device_group = [0]


# compile in noncb ##
qeff_model = QEFFAutoModelForCausalLM(model.cpu().to(torch.float), is_tlm=False, continuous_batching=False)
qpc_path = qeff_model.compile(batch_size=1,
                              prefill_seq_len=seq_len,
                              ctx_len=ctx_len,
                              num_devices=len(device_group),
                              num_cores=16,
                              mxfp6_matmul=True, # False
                              mxint8_kv_cache=True)

# # ## compile in cb ##
# qeff_model = QEFFAutoModelForCausalLM(model.cpu().to(torch.float), is_tlm=False, continuous_batching=True)
# qpc_path = qeff_model.compile(
#     batch_size=1,
#     full_batch_size=full_batch_size,
#     prefill_seq_len=seq_len,
#     ctx_len=ctx_len,
#     num_devices=len(device_group),
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
# )

# run model with QEfficient backend ##
prompts = [
    "Once upon a time",
    # "The gold dollar or gold one @-@ dollar piece was a coin struck as a regular issue by the United States Bureau of the Mint from...",
    # "The Sinclair Scientific Programmable was"
]

qeff_model.generate(
    tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=model_path),
    prompts=prompts,
    device_id=device_group,
)