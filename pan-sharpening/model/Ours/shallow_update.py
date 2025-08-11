class CrossModalFusion(nn.Module):
    """跨模态融合模块，增强MS和PAN特征的互补性"""
    def __init__(self, dim):
        super().__init__()
        self.ms_proj = nn.Conv2d(dim, dim, 1)  # MS特征投影
        self.pan_proj = nn.Conv2d(dim, dim, 1)  # PAN特征投影
        self.attention = nn.Conv2d(2 * dim, 2 * dim, 1)  # 融合注意力
        self.relu = nn.LeakyReLU(0.2)
        # 融合后的精炼模块
        self.refine = EnhancedHinResBlock(2*dim, 2*dim)

    def forward(self, ms_feat, pan_feat):
        # 计算跨模态注意力权重
        combined = torch.cat([ms_feat, pan_feat], dim=1)
        attn = torch.sigmoid(self.attention(combined))
        attn_ms, attn_pan = torch.chunk(attn, 2, dim=1)
        
        # 加权融合
        ms_refined = self.ms_proj(ms_feat) * attn_ms
        pan_refined = self.pan_proj(pan_feat) * attn_pan
        
        # 拼接并精炼
        fused = torch.cat([ms_refined, pan_refined], dim=1)
        return self.refine(fused)
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ChannelAttention(nn.Module):
    """轻量级通道注意力，增强有效特征通道"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """轻量级空间注意力，增强边缘和细节区域"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class EnhancedHinResBlock(nn.Module):
    """改进的HIN残差块，整合注意力机制和混合归一化"""
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True, is_ms_branch=False):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        
        # 主卷积路径
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        
        # 注意力机制（MS分支侧重通道注意力，PAN分支侧重空间注意力）
        if is_ms_branch:
            self.attn = ChannelAttention(out_size)  # MS分支用通道注意力
        else:
            self.attn = SpatialAttention()  # PAN分支用空间注意力
            
        # 混合归一化（保留HIN特性同时增强稳定性）
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        
        # 保留原HIN的分块归一化逻辑
        if self.use_HIN:
            out_1, out_2 = torch.chunk(resi, 2, dim=1)
            resi = torch.cat([self.norm(out_1), out_2], dim=1)
            
        resi = self.relu_2(self.conv_2(resi))
        resi = self.attn(resi)  # 应用注意力增强
        
        return self.identity(x) + resi  # 残差连接

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

class CrossModalFusion(nn.Module):
    """跨模态融合模块，增强MS和PAN特征的互补性"""
    def __init__(self, dim):
        super().__init__()
        self.ms_proj = nn.Conv2d(dim, dim, 1)  # MS特征投影
        self.pan_proj = nn.Conv2d(dim, dim, 1)  # PAN特征投影
        self.attention = nn.Conv2d(2 * dim, 2 * dim, 1)  # 融合注意力
        self.relu = nn.LeakyReLU(0.2)
        # 融合后的精炼模块
        self.refine = EnhancedHinResBlock(2*dim, 2*dim)

    def forward(self, ms_feat, pan_feat):
        # 计算跨模态注意力权重
        combined = torch.cat([ms_feat, pan_feat], dim=1)
        attn = torch.sigmoid(self.attention(combined))
        attn_ms, attn_pan = torch.chunk(attn, 2, dim=1)
        
        # 加权融合
        ms_refined = self.ms_proj(ms_feat) * attn_ms
        pan_refined = self.pan_proj(pan_feat) * attn_pan
        
        # 拼接并精炼
        fused = torch.cat([ms_refined, pan_refined], dim=1)
        return self.refine(fused)


class Net(nn.Module):
    def __init__(self, dim=32, depth=1):
        super().__init__()
        
        base_filter = 32
        self.base_filter = base_filter
        
        # 改进点1: 为MS和PAN设计专属编码器，使用增强型残差块
        self.pan_encoder = nn.Sequential(
            nn.Conv2d(1, base_filter, 3, 1, 1),
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=False),  # PAN分支用空间注意力
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=False),
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=False)
        )
        
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(4, base_filter, 3, 1, 1),
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=True),  # MS分支用通道注意力
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=True),
            EnhancedHinResBlock(base_filter, base_filter, is_ms_branch=True)
        )

        self.stage1_ms = self.ms_encoder
        self.stage1_pan = self.pan_encoder

        # 改进点2: 使用跨模态融合模块替代普通HinResBlock
        # self.stage3_fusion = nn.Sequential(
        #     CrossModalFusion(base_filter),  # 先进行跨模态注意力融合
        #     EnhancedHinResBlock(2*base_filter, 2*base_filter)  # 再用增强块精炼
        # )
        # 修正：使用单个融合模块替代Sequential，支持多输入
        # self.stage3_fusion = CrossModalFusion(base_filter)

        self.stage3_fusion = nn.Sequential(HinResBlock(2*base_filter,2*base_filter))

        # 保持原重建模块，确保输出维度正确
        self.reconstruct = nn.Sequential(
            nn.Conv2d(2*base_filter, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, 1)
        )

    def forward(self, lms, _, pan):
        ms_bic = F.interpolate(lms, scale_factor=4, mode='bilinear', align_corners=False)

        # Stage 1: 使用改进的编码器提取特征
        feat_ms = self.stage1_ms(ms_bic)  # MS特征（侧重光谱）
        feat_pan = self.stage1_pan(pan)   # PAN特征（侧重空间）

        # # Stage 3: 跨模态融合
        # feat = self.stage3_fusion(feat_ms, feat_pan)  # 传入两个分支特征进行融合
        
        B, C, H, W = feat_ms.shape
        # 由于hnetwrapper里就是bchw，这里不需要转换了
        # feat_ms = rearrange(feat_ms, 'b c h w -> b (h w) c')
        # feat_pan = rearrange(feat_pan, 'b c h w -> b (h w) c')

        # Stage 2: 直接将 处理完的图片 使用HNet

        # feat_ms = self.stage2_ms(feat_ms)
        # feat_pan = self.stage2_pan(feat_pan)

        # Stage 3: Cross-Chunk Fusion
        feat = torch.cat([feat_ms, feat_pan], dim=1)
        feat = self.stage3_fusion(feat)

        # # Stage 3: 跨模态融合
        # feat = self.stage3_fusion(feat_ms, feat_pan)  # 传入两个分支特征进行融合


        # 重建输出
        out = self.reconstruct(feat)
        
        return out + ms_bic  # 残差连接，基于双三次插值结果优化


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    lms = torch.randn(1, 4, 32, 32).to(device)  # 4通道MS数据
    pan = torch.randn(1, 1, 128, 128).to(device)  # 1通道PAN数据
    out = model(lms, None, pan)
    print('输出形状:', out.shape)  # 应输出 torch.Size([1, 4, 128, 128])
    print('模型参数数量:', sum(p.numel() for p in model.parameters()))
    
