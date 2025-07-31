import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
import h5py

import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
#from utils.RLD import reConstruct


class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.aspp = nn.ModuleList()
        for rate in dilation_rates:
            self.aspp.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate))
        self.conv1x1 = nn.Conv2d(len(dilation_rates) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        aspp_out = [conv(x) for conv in self.aspp]
        x = torch.cat(aspp_out, dim=1)
        x = self.conv1x1(x)
        return x


class UNet_RLD(nn.Module):
    def __init__(self, lf_extra, n_slices, output_size, channels_interp=128, normalize_mode='percentile'):
        super(UNet_RLD, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(64, 128),
            MultiResBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.decoder = nn.ModuleList([
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # 256 (encoder out) + 256 (previous decoder out)
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # 128 (encoder out) + 128 (previous decoder out)
            nn.Conv2d(64 + 64, n_slices, kernel_size=3, padding=1)  # 64 (encoder out) + 64 (previous decoder out)
        ])
        self.output_size = output_size
        self.normalize_mode = normalize_mode
        with h5py.File('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/PSF_G.mat', 'r') as f:
            PSF = f['PSF_1'][:]

        self.PSF = PSF.astype(np.float32)
        self.PSF = torch.from_numpy(self.PSF).float()
        self.PSF = torch.permute(self.PSF, (2, 1, 0))

    def forward(self, x, x2):
        x = x.permute(0, 2, 1, 3)
        x2 = x2.permute(0, 2, 1, 3)

        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        for i in range(len(enc_outs) - 1, -1, -1):
            x = F.interpolate(x, size=enc_outs[i].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outs[i]], dim=1)
            x = self.decoder[len(enc_outs) - 1 - i](x)

        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        if self.normalize_mode == 'max':
            x = torch.tanh(x)

        x = x.squeeze(0)
        x2 = x2.squeeze(0, 1)
        x = reConstruct(x2.numpy(), x.numpy(), self.PSF.numpy())

        return x




class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, enc_out):
        x = self.upconv(x)
        x = torch.cat([x, enc_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, lf_extra, n_slices, output_size, channels_interp=128, normalize_mode='percentile'):
        super(UNet, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(64, 128),
            MultiResBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.decoder = nn.ModuleList([
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # 256 (encoder out) + 256 (previous decoder out)
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # 128 (encoder out) + 128 (previous decoder out)
            nn.Conv2d(64 + 64, n_slices, kernel_size=3, padding=1)  # 64 (encoder out) + 64 (previous decoder out)
        ])
        self.output_size = output_size
        self.normalize_mode = normalize_mode

    def forward(self, x):
        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        for i in range(len(enc_outs) - 1, -1, -1):
            x = F.interpolate(x, size=enc_outs[i].shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([x, enc_outs[i]], dim=1)
            x = self.decoder[len(enc_outs) - 1 - i](x)


        #这一步实际上没有执行
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        if self.normalize_mode == 'max':
            x = torch.tanh(x)
        return x


class UNet_Transpose(nn.Module):
    def __init__(self, lf_extra, n_slices, output_size, channels_interp=128, normalize_mode='percentile'):
        super(UNet_Transpose, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(64, 128),
            MultiResBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义两个解码器部分，一个用于反卷积，一个用于拼接后的卷积处理
        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 27, kernel_size=4, stride=2, padding=1)  # 调整为27以匹配输入通道数
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(27 + 64, n_slices, kernel_size=3, padding=1)  # 调整通道数
        ])

        self.output_size = output_size
        self.normalize_mode = normalize_mode

    def forward(self, x):
        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        # 从最后一个编码器输出开始逐层进行解码
        for i in range(len(enc_outs) - 1, -1, -1):
            # 反卷积
            x = self.upconv_layers[len(enc_outs) - 1 - i](x)
            # 拼接编码器输出
            x = torch.cat([x, enc_outs[i]], dim=1)
            # 解码器处理
            x = self.decoder_layers[len(enc_outs) - 1 - i](x)


        # 最后的插值到目标输出尺寸，实际不执行，因为输入和输出的尺寸是一致的
        # x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        # if self.normalize_mode == 'max':
        #     x = torch.tanh(x)
        return x
class UNet_VCD(nn.Module):
    def __init__(self, lf_extra=28, n_slices=300, output_size=(600, 600), channels_interp=64, normalize_mode='percentile'):
        super(UNet_VCD, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(64, 128),
            MultiResBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.decoder = nn.ModuleList([
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # 256 (encoder out) + 256 (previous decoder out)
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # 128 (encoder out) + 128 (previous decoder out)
            nn.Conv2d(64 + 64, n_slices, kernel_size=3, padding=1)  # 64 (encoder out) + 64 (previous decoder out)
        ])
        self.output_size = output_size
        self.normalize_mode = normalize_mode

    def forward(self, x):
        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        for i in range(len(enc_outs) - 1, -1, -1):
            x = F.interpolate(x, size=enc_outs[i].shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([x, enc_outs[i]], dim=1)
            x = self.decoder[len(enc_outs) - 1 - i](x)


        #这一步实际上没有执行
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        if self.normalize_mode == 'max':
            x = torch.tanh(x)
        return x


class UNet_Sigmoid(nn.Module):
    def __init__(self, lf_extra, n_slices, output_size, channels_interp=128, normalize_mode='percentile'):
        super(UNet_Sigmoid, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(64, 128),
            MultiResBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义两个解码器部分，一个用于反卷积，一个用于拼接后的卷积处理
        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 27, kernel_size=4, stride=2, padding=1)  # 调整为27以匹配输入通道数
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(27 + 64, n_slices, kernel_size=3, padding=1)  # 调整通道数
        ])

        self.output_size = output_size
        self.normalize_mode = normalize_mode

    def forward(self, x):
        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        # 从最后一个编码器输出开始逐层进行解码
        for i in range(len(enc_outs) - 1, -1, -1):
            # 反卷积
            x = self.upconv_layers[len(enc_outs) - 1 - i](x)
            #print(f"After upconv {len(enc_outs) - 1 - i}, shape: {x.shape}")
            # 拼接编码器输出
            x = torch.cat([x, enc_outs[i]], dim=1)
            #print(f"After concat {i}, shape: {x.shape}")
            # 解码器处理
            x = self.decoder_layers[len(enc_outs) - 1 - i](x)
            #print(f"After decoder {len(enc_outs) - 1 - i}, shape: {x.shape}")
        x = torch.sigmoid(x)
        # 最后的插值到目标输出尺寸
        # x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        # if self.normalize_mode == 'max':
        #     x = torch.tanh(x)

        return x


#--------------------更深的Unet--------------------------------------------------

# 测试网络
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_Deeper(nn.Module):
    def __init__(self, lf_extra=28, n_slices=300, output_size=(600,600), channels_interp=512, normalize_mode='percentile'):
        super(UNet_Deeper, self).__init__()
        self.input_layer = nn.Conv2d(lf_extra, channels_interp, kernel_size=7, padding=3)
        self.aspp = ASPP(channels_interp, channels_interp, [1, 2, 4])
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_interp, 256, kernel_size=3, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            MultiResBlock(256, 512),
            MultiResBlock(512, 1024)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义两个解码器部分，一个用于反卷积，一个用于拼接后的卷积处理
        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 108, kernel_size=4, stride=2, padding=1)  # 调整为108以匹配输入通道数
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(512 + 1024, 512, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),  # 调整通道数
            nn.Conv2d(108 + 256, n_slices, kernel_size=3, padding=1)  # 调整通道数
        ])

        self.output_size = output_size
        self.normalize_mode = normalize_mode

    def forward(self, x):
        x = self.input_layer(x)
        x = self.aspp(x)

        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)

        # 从最后一个编码器输出开始逐层进行解码
        for i in range(len(enc_outs) - 1, -1, -1):
            # 反卷积
            x = self.upconv_layers[len(enc_outs) - 1 - i](x)
            # 拼接编码器输出
            x = torch.cat([x, enc_outs[i]], dim=1)
            # 解码器处理
            x = self.decoder_layers[len(enc_outs) - 1 - i](x)

        # 最后的插值到目标输出尺寸，实际不执行，因为输入和输出的尺寸是一致的
        # x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        # if self.normalize_mode == 'max':
        #     x = torch.tanh(x)
        return x




if __name__ == "__main__":
# Example usage
    lf_extra = 28  # Number of input channels (example)
    n_slices = 300  # Number of output slices
    output_size = (600, 600)  # Output size

    model = UNet_Deeper(lf_extra, n_slices, output_size)
    input_tensor = torch.randn(1, lf_extra, 600, 600)  # Example input tensor
    output = model(input_tensor)
    print(output.shape)