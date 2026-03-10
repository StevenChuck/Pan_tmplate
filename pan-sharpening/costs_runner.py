import os
import time
import torch
import importlib
from thop import profile
from utils.config import get_config


# def run_costs(path_ms, path_pan, path_predict, save_path=None, cfg=None, device='cuda'):
def run_costs(cfg, writer=None):
    """
    一体化运行指标计算 + 模型复杂度评估
    包含 FLOPs / Params / Memory / 推理时间 等

    Args:
        path_ms: str, 多光谱图像路径
        path_pan: str, 全色图像路径
        path_predict: str, 模型输出预测图像路径
        save_path: str, 保存路径 (默认保存到 cfg['test']['save_dir'])
        device: str, 设备 (默认 'cuda')
    """

    path_ms = os.path.join(cfg['data_dir_eval'], 'ms')
    path_pan = os.path.join(cfg['data_dir_eval'], 'pan')
    path_predict = os.path.join(cfg['test']['save_dir'], 'test')
    save_path = cfg['test']['save_dir']
    gpu_id = cfg.get('gpus', [0])[0]
    device = torch.device(f"cuda:{gpu_id}")

    # === 加载配置 ===
    algorithm = cfg['algorithm'].lower()

    # === 动态导入模型 ===
    lib = importlib.import_module('model.' + algorithm)
    Net = lib.Net
     
    model = Net(
        num_channels=cfg['data']['n_colors'], 
            # base_filter=32,
        base_filter=32,
        args = cfg
        
    ).to(device)
    model.eval()

    # === 构造测试输入 ===
    input_ms = torch.randn(1, 4, 32, 32).to(device)
    input_pan = torch.randn(1, 1, 128, 128).to(device)
    input_lms = torch.randn(1, 4, 128, 128).to(device)

    # === 计算显存占用 ===
    torch.cuda.reset_max_memory_allocated(device)
    with torch.no_grad():
        _ = model(input_ms, input_lms, input_pan)
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**3

    # === 计算 FLOPs / Params ===
    flops, params = profile(model, inputs=(input_ms, input_lms, input_pan))
    flops /= 1e9  # GFlops
    params /= 1e6  # MParams

    # === 推理时间计算 ===
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ms, input_lms, input_pan)  # 预热
        start = time.time()
        for _ in range(100):
            _ = model(input_ms, input_lms, input_pan)
        end = time.time()
    avg_time_ms = (end - start) * 1000 / 100

   
    with torch.no_grad():
        model(input_ms, input_lms, input_pan)
        start = time.time()
        for i in range(100):
            model(input_ms, input_lms, input_pan)
        end = time.time()
    elapsed_time_in_seconds = end - start
    elapsed_time_in_milliseconds = elapsed_time_in_seconds * 1000/100

    # === 打印结果 ===
    print(f"\n=== Model Performance Summary ===")
    print(f"Model: {algorithm}")
    print(f"Memory Used: {mem_used:.6f} GB")
    print(f"FLOPs: {flops:.6f} G")
    print(f"Params: {params:.6f} M")
    print(f"Avg Inference Time: {avg_time_ms:.3f} ms")
    print(f"time: ", elapsed_time_in_milliseconds)

    # === 保存结果到文件 ===
    # if save_path is not None:
    #     os.makedirs(save_path, exist_ok=True)
    
        # result_path = os.path.join(save_path, "model_metrics.txt")
    save_path= save_path + 'metrics_result.txt'
    with open(save_path, "a") as f:
        f.write(f"Model: {algorithm}\n")
        f.write(f"Memory Used: {mem_used:.6f} GB\n")
        f.write(f"FLOPs: {flops:.6f} G\n")
        f.write(f"Params: {params:.6f} M\n")
        f.write(f"Avg Inference Time: {avg_time_ms:.3f} ms\n")
        # print("time: ", elapsed_time_in_milliseconds)
        f.write(f"Avg Inference Time: {elapsed_time_in_milliseconds}\n")
        # f.write(f"MS Path: {path_ms}\n")
        # f.write(f"PAN Path: {path_pan}\n")
        f.write(f"Predict Path: {path_predict}\n")
    print(f"✅ Costs saved to: {save_path}")
   
    # 返回成本指标字典，用于 HPARAMS
    costs_dict = {
        'Costs/Memory_GB': mem_used,
        'Costs/FLOPs_G': flops,
        'Costs/Params_M': params,
        'Costs/Time_ms': avg_time_ms
    }
    return {
        'model': algorithm,
        'memory_GB': mem_used,
        'flops_G': flops,
        'params_M': params,
        'time_ms': avg_time_ms,
        'save_path': save_path,
        'costs_dict': costs_dict
    }
