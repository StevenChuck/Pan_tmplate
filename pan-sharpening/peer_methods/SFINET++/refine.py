import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x


class Refine(nn.Module):
    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()
        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(n_feat, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)
        return out


# ===============================================================
# 👇 以下模块为未启用的拓展模块（Rubik、Fourier、CALayer 等），保留作备份
# 如需实验时启用，取消注释即可
# ===============================================================

# class CALayer(nn.Module):
#     def __init__(self, channel, reduction):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#         self.process = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1)
#         )
#
#     def forward(self, x):
#         y = self.process(x)
#         y = self.avg_pool(y)
#         z = self.conv_du(y)
#         return z * y + x


# class FourierShift(nn.Module):
#     def __init__(self, nc, shiftPixel=1):
#         super(FourierShift, self).__init__()
#         self.processReal = nn.Sequential(
#             nn.Conv2d(nc, nc, kernel_size=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, kernel_size=1)
#         )
#         self.processImag = nn.Sequential(
#             nn.Conv2d(nc, nc, kernel_size=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, kernel_size=1)
#         )
#         self.output = nn.Conv2d(nc, nc, kernel_size=1)
#         self.shiftPixel = shiftPixel
#
#     def shift(self, x_real, x_imag):
#         x_realUp, x_realDown, x_realLeft, x_realRight = torch.chunk(x_real, 4, dim=1)
#         x_imagUp, x_imagDown, x_imagLeft, x_imagRight = torch.chunk(x_imag, 4, dim=1)
#         x_realUp = torch.roll(x_realUp, -self.shiftPixel, dims=2)
#         x_realDown = torch.roll(x_realDown, self.shiftPixel, dims=2)
#         x_realLeft = torch.roll(x_realLeft, -self.shiftPixel, dims=3)
#         x_realRight = torch.roll(x_realRight, self.shiftPixel, dims=3)
#         x_imagUp = torch.roll(x_imagUp, -self.shiftPixel, dims=2)
#         x_imagDown = torch.roll(x_imagDown, self.shiftPixel, dims=2)
#         x_imagLeft = torch.roll(x_imagLeft, -self.shiftPixel, dims=3)
#         x_imagRight = torch.roll(x_imagRight, self.shiftPixel, dims=3)
#         return torch.cat([x_realUp, x_realDown, x_realLeft, x_realRight], 1), \
#                torch.cat([x_imagUp, x_imagDown, x_imagLeft, x_imagRight], 1)
#
#     def forward(self, x):
#         x_residual = x
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(x, norm='backward')
#         x_real = x_freq.real
#         x_imag = x_freq.imag
#         x_real, x_imag = self.shift(x_real, x_imag)
#         x_processedReal = self.processReal(x_real)
#         x_processedImag = self.processImag(x_imag)
#         x_out = torch.complex(x_processedReal, x_processedImag)
#         x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
#         x_out = self.output(x_out)
#         return x_out + x_residual


# class RubikCube_multiply(nn.Module):
#     def __init__(self, nc, out, shiftPixel=1, gc=1):
#         super(RubikCube_multiply, self).__init__()
#         self.processC1 = nn.Sequential(
#             nn.Conv2d(gc, gc, 1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.processC2 = nn.Sequential(
#             nn.Conv2d(gc, gc, 1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.processC3 = nn.Sequential(
#             nn.Conv2d(gc, gc, 1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.processC4 = nn.Sequential(
#             nn.Conv2d(gc, gc, 1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.processOutput = nn.Sequential(
#             nn.Conv2d(nc, out, 1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.shiftPixel = shiftPixel
#         self.gc = gc
#         self.split_indexes = (gc, gc, gc, gc, nc - gc * 4)
#
#     def shift_feat(self, x, shiftPixel, g):
#         B, C, H, W = x.shape
#         out = torch.zeros_like(x)
#         out[:, g * 0:g * 1, :, :-shiftPixel] = x[:, g * 0:g * 1, :, shiftPixel:]  # left
#         out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]                    # right
#         out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]                    # up
#         out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]                    # down
#         out[:, g * 4:, :, :] = x[:, g * 4:, :, :]                                 # no shift
#         return out
#
#     def forward(self, x):
#         residual = x
#         x_shifted = self.shift_feat(x, self.shiftPixel, self.gc)
#         c1, c2, c3, c4, x2 = torch.split(x_shifted, self.split_indexes, dim=1)
#         c1_processed = self.processC1(c1)
#         c2_processed = self.processC2(c1_processed * c2)
#         c3_processed = self.processC3(c2_processed * c3)
#         c4_processed = self.processC4(c3_processed * c4)
#         out = torch.cat([c1_processed, c2_processed, c3_processed, c4_processed, x2], dim=1)
#         return self.processOutput(out) + residual


# class RefineRubik(nn.Module):
#     def __init__(self, n_feat, out_channel):
#         super(RefineRubik, self).__init__()
#         self.conv_in = RubikCube_multiply(n_feat, n_feat)
#         self.process = nn.Sequential(ChannelAttention(n_feat, 4))
#         self.conv_last = nn.Sequential(RubikCube_multiply(n_feat, n_feat),
#                                        nn.Conv2d(n_feat, out_channel, 1))
#
#     def forward(self, x):
#         out = self.conv_in(x)
#         out = self.process(out)
#         out = self.conv_last(out)
#         return out


# class RefineShift(nn.Module):
#     def __init__(self, n_feat, out_channel):
#         super(RefineShift, self).__init__()
#         self.conv_in = FourierShift(n_feat)
#         self.process = nn.Sequential(ChannelAttention(n_feat, 4))
#         self.conv_last = nn.Sequential(FourierShift(n_feat),
#                                        nn.Conv2d(n_feat, out_channel, 1))
#
#     def forward(self, x):
#         out = self.conv_in(x)
#         out = self.process(out)
#         out = self.conv_last(out)
#         return out

# ===============================================================
