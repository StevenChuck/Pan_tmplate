#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-20 14:44:14
LastEditTime: 2021-03-22 15:25:02
Description: file content
'''
import  sys,os
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/thop")
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/ptflops")
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/torchsummaryX")

from thop import profile, clever_format
from copy import deepcopy



# try:
#     from ultralytics_thop import profile
# except ImportError:
#     from thop import profile

import importlib, torch
from utils.config import get_config
import math
#from  ptflops import get_model_complexity_info
import time

# =============================
# === 日志函数 ===
# =============================
def write_log(log_path, content):
    """追加写日志到文件"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(content + "\n")
    print(content)  # 同时仍然在控制台打印


if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'pan_inn9' #1.414   0.086    #hmb 57.515606   2.155652
    model_name = 'panmamba_baseline_taylor_ablation_3_order'
    model_name = 'pannet'
    # model_name = 'msdcnn'
    # model_name = 'innformer'
    # model_name = 'SFINET' # nope
    # model_name = 'msddn'
    # model_name = 'panflownet' # 也就是pan_inn9.py
    # model_name = 'sfinet++'
    model_name = 'panmamba'
    # model_name = 'panmamba_baseline_taylor_ablation_3_order'
    # model_name = 'awfln'# model_4b.py
    model_name = 'srppnn'
    model_name = 'phom_ab_cnn'
    # model_name = 'panmamba_baseline_taylor_ablation_3_order'
    # model_name = 'panmamba_baseline_finalversion'
    model_name = 'phom_ab_transformer'
    model_name = 'panmamba_baseline_taylor_ablation_0_order'
    # model_name = 'lformer'
    # model_name = 'hoif'
    # model_name = 'phom_ab_swin'
    model_name = 'panmamba_baseline_taylor_ablation_4_order'
    
    
    
    # model_name = 'panmamba_baseline_taylor_ablation_4_order'
    net_name = model_name.lower()
    lib  = importlib.import_module('model.' + net_name)
    net = lib.Net
    cfg = get_config('/data/b507/gpl/PycharmWorkspace/SOTA/Pansharpening-rfssm/pan-sharpening/option.yml')
    # model = net(
    #     ms_channels=4,
    #     pan_channels=1,
    #     n_feat=16,
    # ).cuda(0)
    # model = net(
    #     ms_channels=4,
    #     pan_channels=1,
    #     n_feat=8
    # ).cuda(0)
    # model = net(
    #         num_channels=4,
    #         base_filter=32,
    #         args=cfg
    # ).to("cuda")
    model = net(
            num_channels=4,
            base_filter=32,
            args=cfg
    ).cuda(0)
    device="cuda:0"
    # model = net(
    scale=1

#     input = torch.randn(1, 4, 32, 32).cuda()
#     input1 = torch.randn(1, 1, 128, 128).cuda()
#     input2 = torch.randn(1, 4, 128, 128).cuda()
    input = torch.randn(1, 4, int(32*scale), int(32*scale)).to(device)
    input1 = torch.randn(1, 1, int(128*scale), int(128*scale)).to(device)
    input2 = torch.randn(1, 4, int(128*scale), int(128*scale)).to(device) 
    model.eval()
    torch.cuda.reset_max_memory_allocated(device)
        # -----------------------------
    # 日志路径
    # -----------------------------
    log_dir = "cost_log"
    log_path = os.path.join(log_dir, f"{model_name}_log.txt")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    write_log(log_path, f"\n===== {model_name} ({timestamp}) =====")

#     mem_before = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        model(input, input2, input1)
        mem_after = torch.cuda.memory_allocated(device)
        max_mem_used_during_forward_pass = torch.cuda.max_memory_allocated(device)

    print(f"Memory used by the model: {max_mem_used_during_forward_pass/1024 ** 3} G")

    # macs, params = get_model_complexity_info(model, ((4, 32, 32), (), (1, 128, 128)),
    #                                          as_strings=True,print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # import torchsummaryX
    # torchsummaryX.summary(model, [input.cpu(), None, input1.cpu()])

    # print("The torchsummary result")
    # from torchsummary import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # summary(model.cuda(), [(4, 32, 32), (), (1, 128, 128)])
    #
    #  构造一份模型副本，仅用于统计
    prof_model = deepcopy(model)
    with torch.no_grad():
        print("The thop result")
        # flops, params = profile(model, inputs=(input, input2, input1))
        flops, params = profile(prof_model, inputs=(input, input2,input1))

        print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e6)))


    import time
    with torch.no_grad():
        model(input, input2, input1)
        start = time.time()
        for i in range(100):
            model(input, input2, input1)
        end = time.time()
    elapsed_time_in_seconds = end - start
    elapsed_time_in_milliseconds = elapsed_time_in_seconds * 1000/100

    print("time: ", elapsed_time_in_milliseconds)

    write_log(log_path, f"Inference Time (ms): {elapsed_time_in_milliseconds:.4f}")

    # -----------------------------
    # 总结
    # -----------------------------
    write_log(log_path, "=" * 40)
    write_log(log_path, f"Summary:")
    write_log(log_path, f"FLOPs: {flops/(1e9):.6f} G, Params: {params/(1e6):.6f} M, "
                        f"Time: {elapsed_time_in_milliseconds:.4f} ms, "
                        f"Memory: {max_mem_used_during_forward_pass/1024 ** 3} GB")
    write_log(log_path, "=" * 40)
    # -----------------------------
    # 释放 GPU 显存
    # -----------------------------
    del model, input, input1, input2
    torch.cuda.empty_cache()
    # write_log(log_path, "[INFO] GPU memory has been released.\n")
