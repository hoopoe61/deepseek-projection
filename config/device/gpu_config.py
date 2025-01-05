from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUConfig:
    name: str
    mem_size_GB: float
    mem_bandwidth_GB_per_sec: float
    FP32_TFLOPS: float
    TF32_TFLOPS: float
    BF16_TFLOPS: float
    FP16_TFLOPS: float
    FP8_TFLOPS: Optional[float]
    INT8_TOPS: float
    pcie_bandwidth_GB_per_sec: float
    # HBD (High bandwidth domain)
    hbd_bandwidth_GB_per_sec: float
    devices_per_hbd: int  # For NVL72, devices_per_hbd = 72, devices_per_node = 4
    devices_per_node: int
    # assume NICs_per_node = devices_per_node
    # Network interface controller (NIC) bandwidth in Gb/s
    rdma_nic_bandwidth_Gbps: float
