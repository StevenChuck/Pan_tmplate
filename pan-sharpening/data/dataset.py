#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-19 20:57:29
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
import h5py
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
    ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
    pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
    bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = pan_image.rotate(180)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug


class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(Data, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.mask_image_filenames = None
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        
        self.scale_resolution = cfg['data'].get('scale_resolution', 1.0)  # 默认不缩放
        
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        
        # 应用 scale_resolution 进行等比例缩放
        if self.scale_resolution != 1.0:
            # 计算目标尺寸
            target_w = int(ms_image.size[0] * self.scale_resolution)
            target_h = int(ms_image.size[1] * self.scale_resolution)
            ms_image = ms_image.resize((target_w, target_h), Image.BICUBIC)
            
            target_w = int(pan_image.size[0] * self.scale_resolution)
            target_h = int(pan_image.size[1] * self.scale_resolution)
            pan_image = pan_image.resize((target_w, target_h), Image.BICUBIC)
        
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image,
                                                                 self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None,data_dir_mask=None):
        super(Data_test, self).__init__()
        print(data_dir_mask)
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.scale_resolution = cfg['data'].get('scale_resolution', 1.0)  # 默认不缩放
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index]) #128
        _, file = os.path.split(self.ms_image_filenames[index])
        
        # 应用 scale_resolution 进行等比例缩放
        if self.scale_resolution != 1.0:
            # 计算目标尺寸
            target_w = int(ms_image.size[0] * self.scale_resolution)
            target_h = int(ms_image.size[1] * self.scale_resolution)
            ms_image = ms_image.resize((target_w, target_h), Image.BICUBIC)
            
            target_w = int(pan_image.size[0] * self.scale_resolution)
            target_h = int(pan_image.size[1] * self.scale_resolution)
            pan_image = pan_image.resize((target_w, target_h), Image.BICUBIC)
        
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor), int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

 # transfer ms instead of lms when no-ref
        # lms_image = ms_image
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_eval(data.Dataset):
    def __init__(self, image_dir, upscale_factor, cfg, transform=None):
        super(Data_eval, self).__init__()
        
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        lms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        lms_image = ms_image.crop((0, 0, lms_image.size[0] // self.upscale_factor * self.upscale_factor, lms_image.size[1] // self.upscale_factor * self.upscale_factor))
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        
        if self.data_augmentation:
            lms_image, pan_image, bms_image, _ = augment(lms_image, pan_image, bms_image)
        
        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        return lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class H5PanSharpeningDataset(data.Dataset):
    def __init__(self, h5_path, cfg):
        super().__init__()
        self.h5_path = h5_path
        self.cfg = cfg
        self.normalize = cfg['data']['normalize']
        self.scale_resolution = cfg['data'].get('scale_resolution', 1.0)  # 默认不缩放

        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['ms'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as f:
            ms = f['ms'][index]
            pan = f['pan'][index]
            lms = f['lms'][index]
            gt = f['gt'][index]

        # 转换为 numpy 数组
        ms = np.array(ms, dtype=np.float32)
        pan = np.array(pan, dtype=np.float32)
        lms = np.array(lms, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        
        # 如果数据范围是 [0, 255]，归一化到 [0, 1]
        # 检查数据范围（假设如果最大值 > 1，则需要归一化）
        if ms.max() > 1.0 or pan.max() > 1.0:
            #GF-2数据范围是 [0, 1023]，归一化到 [0, 1]
            ms = ms / 1023.0
            pan = pan / 1023.0
            lms = lms / 1023.0
            gt = gt / 1023.0
            #             | 卫星          | Value                  |
            # | ----------- | ---------------------- |
            # | WorldView‑3 | **2047** ([GitHub][1]) |[0,2047]
            # | QuickBird   | **2047** ([GitHub][1]) |[0,2047]
            # | GaoFen‑2    | **1023** ([GitHub][1]) |[0,1023]
            # | WorldView‑2 | **2047** ([GitHub][1]) |[0,2047]

            # [1]: https://github.com/XiaoXiao-Woo/PanCollection/blob/main/README.md?plain=1&utm_source=chatgpt.com "PanCollection/README.md at main · XiaoXiao-Woo/PanCollection · GitHub"

        
        # 转换为 torch tensor
        ms = torch.from_numpy(ms).float()
        pan = torch.from_numpy(pan).float()
        lms = torch.from_numpy(lms).float()
        gt = torch.from_numpy(gt).float()

        if self.normalize:
            ms = ms * 2 - 1
            pan = pan * 2 - 1
            lms = lms * 2 - 1
            gt = gt * 2 - 1
        
        #邓剑良里的h5索引 lms是上采样过的，我给他纠正过来
        # train_gf2.h5
        # ├── gt   (19809, 4, 64, 64)
        # ├── lms  (19809, 4, 64, 64)
        # ├── ms   (19809, 4, 16, 16)
        # └── pan  (19809, 1, 64, 64)

        # return ms, pan, bms, gt, str(index)
        
        # 数据映射（根据注释，H5 中的 lms 是上采样过的，需要纠正）
        # H5 格式：ms (低分辨率), pan (全色), lms (上采样后的低分辨率), gt (真值)
        # 项目需要的格式：ms_image (真值), lms_image (低分辨率), pan_image (全色), bms_image (上采样后的低分辨率)
        ms_image = gt      # 真值 (ground truth)
        lms_image = ms    # 低分辨率多光谱 (low resolution MS)
        pan_image = pan   # 全色图像 (panchromatic)
        bms_image = lms   # 上采样后的低分辨率 (bicubic upsampled MS)

        # 应用 scale_resolution 进行等比例缩放
        if self.scale_resolution != 1.0:
            # 计算目标尺寸
            _, _, h, w = ms_image.shape
            target_h = int(h * self.scale_resolution)
            target_w = int(w * self.scale_resolution)
            
            # 对 ms_image (gt) 进行缩放
            ms_image = F.interpolate(ms_image.unsqueeze(0), size=(target_h, target_w), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            
            # 对 pan_image 进行缩放
            pan_image = F.interpolate(pan_image.unsqueeze(0), size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False).squeeze(0)
            
            # 对 bms_image (lms) 进行缩放
            bms_image = F.interpolate(bms_image.unsqueeze(0), size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False).squeeze(0)
            
            # 对 lms_image (ms) 进行缩放 - 低分辨率也要按比例缩放
            _, _, ms_h, ms_w = lms_image.shape
            target_ms_h = int(ms_h * self.scale_resolution)
            target_ms_w = int(ms_w * self.scale_resolution)
            lms_image = F.interpolate(lms_image.unsqueeze(0), size=(target_ms_h, target_ms_w), 
                                     mode='bilinear', align_corners=False).squeeze(0)

        return ms_image, lms_image, pan_image, bms_image, str(index)
