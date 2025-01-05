from dataclasses import dataclass

from config.enum import DataType


optimizer_allowed = ["Adam", "AdamW"]


@dataclass
class TrainOptions:
    num_tokens: int
    num_epoch: int = 1
    global_batch_size: int = 1
    fused_atten: bool = True  # MemoryEfficient/Flash Attention
    causal_mask: bool = False  # Whether to take causal mask into account for SDPA
    optimizer: str = "AdamW"
    use_dtype: DataType = DataType.BF16

    def __post_init__(self):
        assert self.optimizer in optimizer_allowed
