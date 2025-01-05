from dataclasses import dataclass
from typing import Optional

from config.enum import NormType


# Ref:
# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (https://arxiv.org/abs/2405.04434)
# https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
# https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json
@dataclass
class DeepSeekV2:
    name: str = "DeepSeek-V2"
    vocab_size: int = 102400
    max_seq_len: int = 4096
    dim: int = 5120
    moe_intermediate_size: int = 1536
    n_layers: int = 60  # num of total layers
    n_dense_layers: int = 1  # num of non-MoE layers
    n_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64  # per-head dim of the decoupled queries and key
    q_lora_rank: Optional[int] = 1536  # query compression dimension
    kv_lora_rank: int = 512  # KV compression dimension
    bias: bool = False
    ffn_swiglu: bool = True
    norm_type: NormType = NormType.RMS_NORM
    n_experts_shared: int = 2
    n_experts_routed: int = 160
    router_top_k: int = 6


# Ref:
# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
@dataclass
class DeepSeekV2Lite(DeepSeekV2):
    name = "DeepSeek-V2-Lite"
    vocab_size = 102400
    max_seq_len = 4096
    dim = 2048
    moe_intermediate_size = 1408
    n_layers = 27
    n_dense_layers = 1
    n_heads = 16
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_lora_rank = None
    kv_lora_rank = 512
    bias = False
    ffn_swiglu = True
    norm_type = NormType.RMS_NORM
    n_experts_shared = 2
    n_experts_routed = 64
    router_top_k = 6


# Ref: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/config.json
@dataclass
class DeepSeekV3(DeepSeekV2):
    name = "DeepSeek-V3-Base"
    vocab_size = 129280
    max_seq_len = 4096
    dim = 7168
    moe_intermediate_size = 2048
    n_layers = 61
    n_dense_layers = 3
    n_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_lora_rank = 1536
    kv_lora_rank = 512
    bias = False
    ffn_swiglu = True
    norm_type = NormType.RMS_NORM
    n_experts_shared = 1
    n_experts_routed = 256
    router_top_k = 8
