# metrics_runner.py

import os
import re

import cv2
import h5py
import numpy as np
from PIL import Image

from metrics import ref_evaluate, no_ref_evaluate

def cal(ref, noref):
    reflist = []
    noreflist = []
    reflist.append(np.mean([ii[0] for ii in ref]))
    reflist.append(np.mean([ii[1] for ii in ref]))
    reflist.append(np.mean([ii[2] for ii in ref]))
    reflist.append(np.mean([ii[3] for ii in ref]))
    reflist.append(np.mean([ii[4] for ii in ref]))
    reflist.append(np.mean([ii[5] for ii in ref]))
    noreflist.append(np.mean([ih[0] for ih in noref]))
    noreflist.append(np.mean([ih[1] for ih in noref]))
    noreflist.append(np.mean([ih[2] for ih in noref]))
    return reflist, noreflist

def _to_uint8_image(array):
    """将任意张量转换为 HWC 的 uint8 图像。"""
    image = np.array(array, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _collect_prediction_files(path_predict):
    valid_suffix = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    files = []
    for fname in os.listdir(path_predict):
        lower = fname.lower()
        if not lower.endswith(valid_suffix):
            continue
        if lower.endswith('_gt.tif') or lower.endswith('_gt.png') or lower.endswith('_gt.jpg'):
            continue
        files.append(fname)
    files.sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', x)])
    return files


def _evaluate_from_directories(path_ms, path_pan, path_predict, scale_resolution=1.0):
    list_name = os.listdir(path_ms)
    list_ref = []
    list_noref = []

    for file_name_i in list_name:
        path_ms_file = os.path.join(path_ms, file_name_i)
        path_pan_file = os.path.join(path_pan, file_name_i)
        path_predict_file = os.path.join(path_predict, file_name_i)

        gt_name = file_name_i.split('.')[0] + '_gt.tif'
        gt_path_file = os.path.join(path_predict, gt_name)
        
        gt = np.uint8(np.array(Image.open(gt_path_file)))#这里使用的是原始数据的高分辨率MS图像
        original_msi = np.array(Image.open(path_ms_file))
        original_pan = np.array(Image.open(path_pan_file))
        fused_image = np.array(Image.open(path_predict_file))
        
       

        
        

        # gt = np.uint8(original_msi)
        scale = int(original_msi.shape[0]/fused_image.shape[0])#这里是通过原始数据HRMS和模型输出的patch后的HRMS相除得到的比例因子，保证后面的指标函数正常运行
        used_ms = cv2.resize(original_msi, (original_msi.shape[1] // (scale*4), original_msi.shape[0] // (scale*4)), cv2.INTER_CUBIC)
        used_pan = cv2.resize(original_pan, (original_pan.shape[1] // scale, original_pan.shape[0] // scale), cv2.INTER_CUBIC)
        used_pan = np.expand_dims(used_pan, -1)

        list_ref.append(ref_evaluate(fused_image, gt))
        list_noref.append(no_ref_evaluate(fused_image, np.uint8(used_pan), np.uint8(used_ms)))

    return cal(list_ref, list_noref)


def _evaluate_from_h5(h5_path, path_predict, scale_resolution=1.0):
    list_ref = []
    list_noref = []

    predict_files = _collect_prediction_files(path_predict)
    if not predict_files:
        raise FileNotFoundError(f"未在 {path_predict} 中找到预测图像，请先运行推理。")

    with h5py.File(h5_path, 'r') as h5_file:
        gt_ds = h5_file['gt']
        pan_ds = h5_file['pan']

        for pred_name in predict_files:
            index_part = os.path.splitext(pred_name)[0]
            match = re.search(r'\d+', index_part)
            if match is None:
                raise ValueError(f"预测文件 {pred_name} 无法解析出索引，请使用数字索引命名。")
            idx = int(match.group())
            if idx >= gt_ds.shape[0]:
                raise IndexError(f"预测索引 {idx} 超出 H5 数据集长度 {gt_ds.shape[0]}")

            fused_image = np.array(Image.open(os.path.join(path_predict, pred_name)))

            # 从 H5 读取原始数据
            gt_raw = gt_ds[idx]
            pan_raw = pan_ds[idx]
            
            # 应用 scale_resolution 缩放
            if scale_resolution != 1.0:
                # 计算目标尺寸
                if gt_raw.ndim == 3:  # (C, H, W)
                    _, h, w = gt_raw.shape
                    target_h = int(h * scale_resolution)
                    target_w = int(w * scale_resolution)
                    # 转换为 HWC 格式进行缩放
                    gt_raw_hwc = np.transpose(gt_raw, (1, 2, 0))
                    gt_raw_hwc = cv2.resize(gt_raw_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    gt_raw = np.transpose(gt_raw_hwc, (2, 0, 1))
                else:  # (H, W, C) 或其他格式
                    h, w = gt_raw.shape[:2]
                    target_h = int(h * scale_resolution)
                    target_w = int(w * scale_resolution)
                    gt_raw = cv2.resize(gt_raw, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                
                # 对 pan 进行缩放
                if pan_raw.ndim == 3:  # (C, H, W)
                    _, h, w = pan_raw.shape
                    target_h = int(h * scale_resolution)
                    target_w = int(w * scale_resolution)
                    pan_raw_hwc = np.transpose(pan_raw, (1, 2, 0))
                    pan_raw_hwc = cv2.resize(pan_raw_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    pan_raw = np.transpose(pan_raw_hwc, (2, 0, 1))
                else:  # (H, W) 或 (H, W, C)
                    h, w = pan_raw.shape[:2]
                    target_h = int(h * scale_resolution)
                    target_w = int(w * scale_resolution)
                    pan_raw = cv2.resize(pan_raw, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            gt = _to_uint8_image(gt_raw)
            used_ms = cv2.resize(gt, (gt.shape[1] // 4, gt.shape[0] // 4), cv2.INTER_CUBIC)

            pan = _to_uint8_image(pan_raw)
            if pan.ndim == 3 and pan.shape[2] == 1:
                used_pan = pan
            elif pan.ndim == 2:
                used_pan = pan[..., None]
            else:
                used_pan = pan[..., :1]

            list_ref.append(ref_evaluate(fused_image, gt))
            list_noref.append(no_ref_evaluate(fused_image, np.uint8(used_pan), np.uint8(used_ms)))

    return cal(list_ref, list_noref)


def run_metrics(path_ms, path_pan, path_predict, save_path='', cfg=None):
    metrics_path = os.path.join(save_path, 'metrics_result.txt')
    
    # 从配置中获取 scale_resolution，默认为 1.0（不缩放）
    scale_resolution = 1.0
    if cfg is not None and 'data' in cfg and 'scale_resolution' in cfg['data']:
        scale_resolution = cfg['data']['scale_resolution']

    if path_ms.lower().endswith('.h5'):
        temp_ref_results1, temp_no_ref_results1 = _evaluate_from_h5(path_ms, path_predict, scale_resolution)
    else:
        temp_ref_results1, temp_no_ref_results1 = _evaluate_from_directories(path_ms, path_pan, path_predict, scale_resolution)

    ref_results = {'metrics: ': '  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q', 'deep': temp_ref_results1}
    no_ref_results = {'metrics: ': '  D_lamda,  D_s,    QNR', 'deep': temp_no_ref_results1}

    # 保存到 TXT
    with open(metrics_path, 'w') as f:
        f.write('################## reference comparision #######################\n')
        for index, i in enumerate(ref_results):
            if index == 0:
                f.write(f"{i}: {ref_results[i]}\n")
            else:
                f.write(f"{i}: {[round(j, 4) for j in ref_results[i]]}\n")

        f.write('################## no reference comparision ####################\n')
        for index, i in enumerate(no_ref_results):
            if index == 0:
                f.write(f"{i}: {no_ref_results[i]}\n")
            else:
                f.write(f"{i}: {[round(j, 4) for j in no_ref_results[i]]}\n")

    print(f"✔ 指标结果已保存至：{metrics_path}")

