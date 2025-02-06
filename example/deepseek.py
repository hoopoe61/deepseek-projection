from config.device.nv_gpu import H800SXM80GB
from config.enum import DataType
from config.train_options import TrainOptions

from config.model.deepseek_config import DeepSeekV2, DeepSeekV2Lite, DeepSeekV3, Nanbeige330B, Nanbeige450B
from projection.deepseek_proj import DeepSeekProjection


def project_330B():
    TOKENS_V3 = 10 * 10**12 #训练了多少数据
    GBS_V3 = 15360
    #GPU_HOURS_V3 = 180 * 10**3 * 14.8  # K GPU Hours/T Tokens #训练1T token所需要的时间
    #GPU_HOURS_V3 = 2.788 * 10**6 #来自report报告

    model_config = Nanbeige330B
    print("-" * 50)
    print(f"Model: {model_config.name}")

    train_options = TrainOptions(
        num_tokens=TOKENS_V3,
        global_batch_size=GBS_V3,
        causal_mask=True,
        use_dtype=DataType.FP8,
    )
    proj = DeepSeekProjection(model_config, train_options)

    num_sparse_params, num_activated_params = proj.get_num_params()
    print(
        f"总参数量: {num_sparse_params / 10 ** 9:.2f}B, 激活参数量: {num_activated_params / 10 ** 9:.2f}B"
    )

    gpu_config = H800SXM80GB
    num_flop_per_token = proj.get_num_flop_per_token()
    #print(f"FLOP_per_token: {num_flop_per_token}, TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    total_flops_in_tera = (num_flop_per_token * TOKENS_V3) / 10**12
    #print(f"TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    gpu_tflops = gpu_config.FP8_TFLOPS
    
    print(f"每个token所使用的FLOP: {num_flop_per_token}, 每个token所使用的TFLOP: {num_flop_per_token / 10 ** 12:.4f}")
    
    print(f"token总数量: {TOKENS_V3}, 训练这些token所需的总TFLOP: {total_flops_in_tera}")
    
    gpu_hours = total_flops_in_tera/gpu_tflops/3600.0
    print(f"H800 fp8(稠密算力)TFLOPS: {gpu_tflops}, 理论上的所需的H800小时：{gpu_hours}, 按照每台机器8卡，理论上的所需的H800机器小时：{gpu_hours/8.0}, 理论上的所需的H800机器天：{gpu_hours/8.0/24.0}")
    
    MFU = 0.15
    print(f"假定MFU是: {MFU}，那么所需的H800小时：{gpu_hours/MFU}, 按照每台机器8卡，所需的H800机器小时：{gpu_hours/8.0/MFU}, 所需的H800机器天：{gpu_hours/8.0/24.0/MFU}")
    #mfu_in_percent = total_flops_in_tera / (gpu_tflops * GPU_HOURS_V3 * 3600) * 100
    #print(f"Calculated MFU with GPU hours from paper: {mfu_in_percent:.2f}%")

def project_450B():
    TOKENS_V3 = 10 * 10**12 #训练了多少数据
    GBS_V3 = 15360
    #GPU_HOURS_V3 = 180 * 10**3 * 14.8  # K GPU Hours/T Tokens #训练1T token所需要的时间
    #GPU_HOURS_V3 = 2.788 * 10**6 #来自report报告

    model_config = Nanbeige450B
    print("-" * 50)
    print(f"Model: {model_config.name}")

    train_options = TrainOptions(
        num_tokens=TOKENS_V3,
        global_batch_size=GBS_V3,
        causal_mask=True,
        use_dtype=DataType.FP8,
    )
    proj = DeepSeekProjection(model_config, train_options)

    num_sparse_params, num_activated_params = proj.get_num_params()
    print(
        f"总参数量: {num_sparse_params / 10 ** 9:.2f}B, 激活参数量: {num_activated_params / 10 ** 9:.2f}B"
    )

    gpu_config = H800SXM80GB
    num_flop_per_token = proj.get_num_flop_per_token()
    #print(f"FLOP_per_token: {num_flop_per_token}, TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    total_flops_in_tera = (num_flop_per_token * TOKENS_V3) / 10**12
    #print(f"TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    gpu_tflops = gpu_config.FP8_TFLOPS
    
    print(f"每个token所使用的FLOP: {num_flop_per_token}, 每个token所使用的TFLOP: {num_flop_per_token / 10 ** 12:.4f}")
    
    print(f"token总数量: {TOKENS_V3}, 训练这些token所需的总TFLOP: {total_flops_in_tera}")
    
    gpu_hours = total_flops_in_tera/gpu_tflops/3600.0
    print(f"H800 fp8(稠密算力)TFLOPS: {gpu_tflops}, 理论上的所需的H800小时：{gpu_hours}, 按照每台机器8卡，理论上的所需的H800机器小时：{gpu_hours/8.0}, 理论上的所需的H800机器天：{gpu_hours/8.0/24.0}")
    
    MFU = 0.15
    print(f"假定MFU是: {MFU}，那么所需的H800小时：{gpu_hours/MFU}, 按照每台机器8卡，所需的H800机器小时：{gpu_hours/8.0/MFU}, 所需的H800机器天：{gpu_hours/8.0/24.0/MFU}")
    #mfu_in_percent = total_flops_in_tera / (gpu_tflops * GPU_HOURS_V3 * 3600) * 100
    #print(f"Calculated MFU with GPU hours from paper: {mfu_in_percent:.2f}%")

def project_v3():
    TOKENS_V3 = 14.8 * 10**12 #训练了多少数据
    GBS_V3 = 15360
    #GPU_HOURS_V3 = 180 * 10**3 * 14.8  # K GPU Hours/T Tokens #训练1T token所需要的时间
    GPU_HOURS_V3 = 2.788 * 10**6 #来自report报告

    model_config = DeepSeekV3
    print("-" * 50)
    print(f"Model: {model_config.name}")

    train_options = TrainOptions(
        num_tokens=TOKENS_V3,
        global_batch_size=GBS_V3,
        causal_mask=True,
        use_dtype=DataType.FP8,
    )
    proj = DeepSeekProjection(model_config, train_options)

    num_sparse_params, num_activated_params = proj.get_num_params()
    print(
        f"总参数量: {num_sparse_params / 10 ** 9:.2f}B, 激活参数量: {num_activated_params / 10 ** 9:.2f}B"
    )

    gpu_config = H800SXM80GB
    num_flop_per_token = proj.get_num_flop_per_token()
    #print(f"FLOP_per_token: {num_flop_per_token}, TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    total_flops_in_tera = (num_flop_per_token * TOKENS_V3) / 10**12
    #print(f"TFLOP_per_token: {num_flop_per_token / 10 ** 12:.4f}")
    gpu_tflops = gpu_config.FP8_TFLOPS
    
    print(f"每个token所使用的FLOP: {num_flop_per_token}, 每个token所使用的TFLOP: {num_flop_per_token / 10 ** 12:.4f}")
    
    print(f"token总数量: {TOKENS_V3}, 训练这些token所需的总TFLOP: {total_flops_in_tera}")
    
    print(f"所需的H800小时：{GPU_HOURS_V3}, 按照每台机器8卡，所需的H800机器小时：{GPU_HOURS_V3/8.0}, 所需的H800机器天：{GPU_HOURS_V3/8.0/24.0}")
    
    mfu_in_percent = total_flops_in_tera / (gpu_tflops * GPU_HOURS_V3 * 3600) * 100
    print(f"按照report中数据计算出来的MFU: {mfu_in_percent:.2f}%")

def project_v2():
    TOKENS_V2 = 8.1 * 10**12
    GBS_V2 = 9216
    GPU_HOURS_V2 = 172.8 * 10**3 * 8.1  # K GPU Hours/T Tokens

    model_config = DeepSeekV2
    print("-" * 50)
    print(f"Model: {model_config.name}")

    train_options = TrainOptions(
        num_tokens=TOKENS_V2, global_batch_size=GBS_V2, causal_mask=True
    )
    proj = DeepSeekProjection(model_config, train_options)

    num_sparse_params, num_activated_params = proj.get_num_params()
    print(
        f"num_sparse_params: {num_sparse_params / 10**9:.2f}B, num_activated_params: {num_activated_params / 10**9:.2f}B"
    )

    gpu_config = H800SXM80GB
    num_flop_per_token = proj.get_num_flop_per_token()
    print(f"TFLOP_per_token: {num_flop_per_token / 10**12:.4f}")
    total_flops_in_tera = (num_flop_per_token * TOKENS_V2) / 10**12
    mfu_in_percent = (
        total_flops_in_tera / (gpu_config.BF16_TFLOPS * GPU_HOURS_V2 * 3600) * 100
    )
    print(f"Calculated MFU with GPU hours from paper: {mfu_in_percent:.2f}%")


def project_v2_lite():
    TOKENS_V2_LITE = 5.7 * 10**12
    GBS_V2_LITE = 4608

    model_config = DeepSeekV2Lite
    print("-" * 50)
    print(f"Model: {model_config.name}")

    train_options = TrainOptions(
        num_tokens=TOKENS_V2_LITE, global_batch_size=GBS_V2_LITE, causal_mask=True
    )
    proj = DeepSeekProjection(model_config, train_options)

    num_sparse_params, num_activated_params = proj.get_num_params()
    print(
        f"num_sparse_params: {num_sparse_params / 10**9:.2f}B, num_activated_params: {num_activated_params / 10**9:.2f}B"
    )

    num_flop_per_token = proj.get_num_flop_per_token()
    print(f"TFLOP_per_token: {num_flop_per_token / 10**12:.4f}")


if __name__ == "__main__":
    project_v3()
    print("\n\n")
    project_330B()
    print("\n\n")
    project_450B()
