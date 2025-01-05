from dataclasses import dataclass

from config.device.gpu_config import GPUConfig


# config of NVIDIA GPU
@dataclass
class A100SXM80GB(GPUConfig):
    name = "A100-SXM-80GB"
    mem_size_GB = 80
    mem_bandwidth_GB_per_sec = 2039
    FP32_TFLOPS = 19.5
    TF32_TFLOPS = 156
    BF16_TFLOPS = 312
    FP16_TFLOPS = 312
    INT8_TOPS = 624
    pcie_bandwidth_GB_per_sec = 64
    hbd_bandwidth_GB_per_sec = 600  # NVLink
    devices_per_hbd = 8  # NVLink domain size
    devices_per_node = 8
    rdma_nic_bandwidth_Gbps = 200


@dataclass
class A800SXM80GB(A100SXM80GB):
    name = "A800-SXM-80GB"
    hbd_bandwidth_GB_per_sec = 400  # NVLink


@dataclass
class H100SXM80GB(GPUConfig):
    name = "H100-SXM-80GB"
    mem_size_GB = 80
    mem_bandwidth_GB_per_sec = 3350
    FP32_TFLOPS = 67
    TF32_TFLOPS = 495
    BF16_TFLOPS = 989
    FP16_TFLOPS = 989
    FP8_TFLOPS = 1979
    INT8_TOPS = 1979
    pcie_bandwidth_GB_per_sec = 128
    hbd_bandwidth_GB_per_sec = 900  # NVLink
    devices_per_hbd = 8
    devices_per_node = 8
    rdma_nic_bandwidth_Gbps = 400


@dataclass
class H800SXM80GB(H100SXM80GB):
    name = "H800-SXM-80GB"
    hbd_bandwidth_GB_per_sec = 400  # NVLink


@dataclass
class H200SXM141GB(H100SXM80GB):
    name = "H200-SXM-141GB"
    mem_size_GB = 141
    mem_bandwidth_GB_per_sec = 4800


# Ref: NVIDIA Blackwell Architecture Technical Brief (https://resources.nvidia.com/en-us-blackwell-architecture)
@dataclass
class B200SXM180GB(GPUConfig):
    name = "B200-SXM-180GB"
    mem_size_GB = 180
    mem_bandwidth_GB_per_sec = 8000
    FP32_TFLOPS = 80
    TF32_TFLOPS = 1100
    BF16_TFLOPS = 2250
    FP16_TFLOPS = 2250
    FP8_TFLOPS = 4500
    INT8_TOPS = 4500
    pcie_bandwidth_GB_per_sec = 128  # PCIe 5.0 (Intel® Xeon® Platinum 8570 Processors)
    hbd_bandwidth_GB_per_sec = 1800
    devices_per_hbd = 8
    devices_per_node = 8
    rdma_nic_bandwidth_Gbps = 400


@dataclass
class GB200(GPUConfig):
    name = "GB200"
    mem_size_GB = 192
    mem_bandwidth_GB_per_sec = 8000
    FP32_TFLOPS = 90
    TF32_TFLOPS = 1250
    BF16_TFLOPS = 2500
    FP16_TFLOPS = 2500
    FP8_TFLOPS = 5000
    INT8_TOPS = 5000
    pcie_bandwidth_GB_per_sec = 256  # PCIe 6.0
    hbd_bandwidth_GB_per_sec = 1800
    devices_per_hbd = 72  # NVL72
    devices_per_node = 4
    rdma_nic_bandwidth_Gbps = 800  # ConnectX-8
