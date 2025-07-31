import os

import math
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import tifffile as tiff
import h5py
# 目标模块路径
nafnet_arch_path = '/gpfs/home/LifeSci/wenlab/hefengcs/models/NAFNet/basicsr/models/archs/'

# 将目标模块路径添加到 Python 搜索路径中
sys.path.append(nafnet_arch_path)
import NAFNet_arch
from NAFNet_arch import NAFNet

#导入Restormer
restorer_arch_path = '/gpfs/home/LifeSci/wenlab/hefengcs/models/Restormer-main/basicsr/models/archs/'
sys.path.append(restorer_arch_path)

PSUNET_path ='/gpfs/home/LifeSci/wenlab/hefengcs/models/PUSNet-main/models'
sys.path.append('/gpfs/home/LifeSci/wenlab/hefengcs/models/PUSNet-main/')
import config as c
sys.path.append(PSUNET_path)
#from utils.image import quantization
from PUSNet import pusnet

# import restormer_arch
from restormer_arch import Restormer
from model.swin_transformer import SwinTransformer

class DownsampleHead(nn.Module):
    def __init__(self):
        super(DownsampleHead, self).__init__()
        self.conv1 = nn.Conv2d(27, 128, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv2 = nn.Conv2d(128, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
class DownsampleHead_location(nn.Module):
    def __init__(self):
        super(DownsampleHead_location, self).__init__()
        self.conv1 = nn.Conv2d(28, 128, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv2 = nn.Conv2d(128, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DeepConvNetwork(nn.Module):
    def __init__(self):
        super(DeepConvNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 300, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



class DownsampleHead_PSF(nn.Module):
    def __init__(self):
        super(DownsampleHead_PSF, self).__init__()
        self.conv1 = nn.Conv2d(27, 128, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv2 = nn.Conv2d(128, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)
        psf_path ='/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_G_600.tif'
        self.conv3 = nn.Conv2d(600, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv4 = nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)
        # Read and register the PSF as a buffer (so it is not treated as a learnable parameter)
        psf = tiff.imread(psf_path).astype(np.float32)  # Make sure PSF is float32 for tensor compatibility
        psf = torch.from_numpy(psf)  # Convert PSF to tensor
        psf = psf.unsqueeze(0)  # Add batch dimension: (1, 300, 600, 600)
        self.register_buffer('psf', psf)  # Register as a buffer, not a parameter

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Apply the PSF by pixel-wise addition
        #x = x + self.psf
        #进行concatenate
        x=torch.cat((x,self.psf),dim=1)

        x =F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))



        return x



class DownsampleHead_deeper(nn.Module):
    def __init__(self):
        super(DownsampleHead_deeper, self).__init__()
        self.conv1 = nn.Conv2d(27, 64, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 300, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x




class RichardsonLucySubModule(torch.nn.Module):
    def __init__(self, PSF, device_id):
        super().__init__()
        #self.init = init.to(device_id)
        #self.PSF = PSF.to(device_id)
        self.device_id = device_id
        self.PSF = PSF

    def forward(self, imstack, init):

        ItN = 2  # 迭代次数
        BkgMean = 110  # 背景噪声平均水平
        ROISize = 300  # 感兴趣区域大小
        SNR = 200  # 信噪比
        NxyExt = 128  # 图像扩展大小
        Nxy = 2304  # 扩展后图像大小
        Nz = 300  # Z轴高度

        PSF = torch.nn.functional.pad(self.PSF, (0, 0, NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)
        gpuObjReconTmp = torch.zeros((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        ImgEst = torch.zeros((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        Ratio = torch.ones((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        Img = imstack.to(self.device_id)
        ImgMultiView = Img - BkgMean
        ImgMultiView = torch.clamp(ImgMultiView, min=0)
        ImgExp = torch.nn.functional.pad(ImgMultiView, (NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

        gpuObjRecon = init.clone()
        gpuObjRecon =gpuObjRecon.to(self.device_id)
        for ii in range(ItN):
            ImgEst.fill_(0)
            for jj in range(Nz):
                gpuObjReconTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                Nxy // 2 - ROISize:Nxy // 2 + ROISize] = gpuObjRecon[jj, :, :]

                PSF_jj = PSF[:, :, jj]
                fft_PSF = torch.fft.fft2(torch.fft.ifftshift(PSF_jj))
                fft_PSF = fft_PSF.to(self.device_id)
                PSF_jj = PSF_jj.to(self.device_id)
                ImgEst = ImgEst.to(self.device_id)
                ImgEst_slice_update = torch.real(torch.fft.ifft2(torch.fft.fft2(gpuObjReconTmp) * fft_PSF)) / torch.sum(
                    PSF_jj)
                ImgEst += torch.max(ImgEst_slice_update, torch.tensor(0.0, device=ImgEst_slice_update.device))

            ImgExpEst = ImgExp

            Tmp = torch.median(ImgEst)

            Ratio.fill_(1)
            Ratio[NxyExt:-NxyExt, NxyExt:-NxyExt] = ImgExpEst[NxyExt:-NxyExt, NxyExt:-NxyExt] / (
                    ImgEst[NxyExt:-NxyExt, NxyExt:-NxyExt] + Tmp / SNR)

            for jj in range(Nz):
                PSF_jj = PSF[:, :, jj]

                PSF_jj_shifted = torch.fft.ifftshift(PSF_jj)

                fft_PSF = torch.fft.fft2(PSF_jj_shifted)

                fft_PSF_conj = torch.conj(fft_PSF)

                fft_Ratio = torch.fft.fft2(Ratio)
                fft_Ratio = fft_Ratio.to(self.device_id)
                fft_PSF_conj = fft_PSF_conj.to(self.device_id)
                ifft_result = torch.fft.ifft2(fft_Ratio * fft_PSF_conj)

                gpuTmp = torch.real(ifft_result) / torch.sum(PSF_jj)

                gpuTmp = torch.maximum(gpuTmp, torch.tensor(0.0, device=self.device_id))

                gpuObjRecon[jj, :, :] *= gpuTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                                         Nxy // 2 - ROISize:Nxy // 2 + ROISize]
        #这里恢复到了600，600，300的尺寸，因此能够引入一个网络来进行训练。



        #ObjRecon = gpuObjRecon.cpu().numpy().astype(np.float32)
        return gpuObjRecon

class NARLD(torch.nn.Module):
    def __init__(self,  device_id):
        super().__init__()
        #self.init = init.to(device_id)

        self.first_model = None
        self.first_model = CombinedModel().to(device_id)
        model_ckpt = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt/NAFNet_pure_PSNR_RLD100_paired900_081220240812-042424/epoch=249-val_loss=0.00.ckpt'  # 指定模型权重文件路径

        # 加载 PyTorch Lightning 检查点并去除前缀
        checkpoint = torch.load(model_ckpt, map_location=device_id)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        self.first_model.load_state_dict(new_state_dict)


        # with h5py.File('/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_G.mat', 'r') as f:
        #     PSF = f['PSF_1'][:]
        # PSF = PSF.astype(np.float32)
        # PSF = np.transpose(PSF, (2, 1, 0))
        # PSF = torch.from_numpy(PSF)
        # self.PSF = nn.Parameter(PSF.to(device_id), requires_grad=True)
        self.PSF =torch.randn( 2048, 2048, 300)
        #self.device_id = device_id
        self.second_model = RichardsonLucySubModule(self.PSF,device_id)


    def forward(self, imstack, crop_image):
        init = self.first_model(crop_image)
        #移除第一个维度
        init = init.squeeze(0)
        recon_image =self.second_model(imstack, init)

        return recon_image



class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.downsample_head = DownsampleHead()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        x = self.relu(x)
        return x
class CombinedModel_location(nn.Module):
    def __init__(self):
        super(CombinedModel_location, self).__init__()
        self.downsample_head = DownsampleHead_location()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        return x

class CombinedModel_deeper(nn.Module):
    def __init__(self):
        super(CombinedModel_deeper, self).__init__()
        self.downsample_head = DownsampleHead_location()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 300
        width = 32

        enc_blks = [2, 2, 2, 2, 2,4]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2, 2,2]

        # enc_blks = [1, 1, 1, 28]
        # middle_blk_num = 1
        # dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        return x

class CombinedModel2(nn.Module):
    def __init__(self):
        super(CombinedModel2, self).__init__()
        #self.downsample_head = DownsampleHead()
        #self.SEBottleneck = SEBottleneck()
        self.downsample_head =DeepConvNetwork()

        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    def forward(self, x):
        x = self.downsample_head(x)


        x = self.restoration_net(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        out = torch.mean(x, dim=(2, 3), keepdim=True)  # 全局平均池化
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out  # 通道注意力

class SEBottleneck(nn.Module):
    def __init__(self):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(27, 64, kernel_size=1)
        self.se1 = SEBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.se2 = SEBlock(128)
        self.conv3 = nn.Conv2d(128, 300, kernel_size=1)
        self.se3 = SEBlock(300)
        self.shortcut = nn.Conv2d(27, 300, kernel_size=1)  # 残差连接

    def forward(self, x):
        residual = self.shortcut(x)  # 残差分支
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = F.relu(self.conv3(x))
        x = self.se3(x)
        return x + residual  # 残差连接
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class CombinedModel_PSF(nn.Module):
    def __init__(self):
        super(CombinedModel_PSF, self).__init__()
        self.downsample_head = DownsampleHead_PSF()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 300
        width = 300

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        return x
class ImageResizerLow(nn.Module):
    def __init__(self):
        super(ImageResizerLow, self).__init__()

        # 使用一个卷积层从 300 通道减少到 1 通道
        self.conv = nn.Conv2d(300, 1, kernel_size=3, stride=1, padding=1)  # (1, 1, 600, 600)

        # 上采样层 1: 从 600x600 到 1024x1024
        self.upsample1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # (1, 1, 1200, 1200)
        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # (1, 1, 2400, 2400)

    def forward(self, x):
        # 输入维度: (1, 300, 600, 600)

        # 使用一个卷积层进行通道压缩
        x = F.relu(self.conv(x))  # (1, 1, 600, 600)

        # 上采样部分
        x = F.relu(self.upsample1(x))  # (1, 1, 1200, 1200)
        x = F.relu(self.upsample2(x))  # (1, 1, 2400, 2400)

        # 最终裁剪到 (1, 1, 2048, 2048)
        x = x[:, :, :2048, :2048]

        return x

class ImageResizer(nn.Module):
    def __init__(self):
        super(ImageResizer, self).__init__()

        # 卷积层 1: 从 300 通道减少到 128 通道
        self.conv1 = nn.Conv2d(300, 128, kernel_size=3, stride=1, padding=1)  # (1, 128, 600, 600)
        self.bn1 = LayerNorm2d(128)

        # 卷积层 2: 从 128 通道减少到 64 通道
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # (1, 64, 600, 600)
        self.bn2 = LayerNorm2d(64)

        # 卷积层 3: 从 64 通道减少到 32 通道
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # (1, 32, 600, 600)
        self.bn3 = LayerNorm2d(32)

        # 卷积层 4: 从 32 通道减少到 16 通道
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # (1, 16, 600, 600)
        self.bn4 = LayerNorm2d(16)

        # 卷积层 5: 从 16 通道减少到 1 通道
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # (1, 1, 600, 600)

        # 上采样层 1: 从 600x600 到 1024x1024
        self.upsample1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # (1, 1, 1200, 1200)
        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # (1, 1, 2400, 2400)

        # 最终裁剪得到 2048x2048
        self.crop = nn.Identity()  # 如果你需要精确裁剪图像

    def forward(self, x):
        # 输入维度: (1, 300, 600, 600)

        # 特征提取部分
        x = F.relu(self.bn1(self.conv1(x)))  # (1, 128, 600, 600)
        x = F.relu(self.bn2(self.conv2(x)))  # (1, 64, 600, 600)
        x = F.relu(self.bn3(self.conv3(x)))  # (1, 32, 600, 600)
        x = F.relu(self.bn4(self.conv4(x)))  # (1, 16, 600, 600)
        x = F.relu(self.conv5(x))  # (1, 1, 600, 600)

        # 上采样部分
        x = F.relu(self.upsample1(x))  # (1, 1, 1200, 1200)
        x = F.relu(self.upsample2(x))  # (1, 1, 2400, 2400)

        # 最终裁剪到 (1, 1, 2048, 2048)
        x = x[:, :, :2048, :2048]

        return x

class CombinedModel_deeper_muti(nn.Module):
    def __init__(self,PSF_fft):
        super(CombinedModel_deeper_muti, self).__init__()
        self.downsample_head = DownsampleHead_location()

        img_channel = 300
        width = 64

        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]

        # 初始化 restoration_net（共享权重）
        self.restoration_net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        # coordinates_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data_prepare/coordinates.txt'  # 坐标信息文件路径
        #
        # # 生成一个随机的 3D 张量作为输入
        # self.relu = nn.ReLU()
        # self.lenses_info = []
        # with open(coordinates_path, 'r') as file:
        #     for line in file:
        #         x, y, w, h = map(int, line.split())
        #         self.lenses_info.append((x, y, w, h))



        #获取设备：self.restoration_net
        self.ImageResizer = ImageResizer()
        #self.PSF_fft = PSF_fft
        self.register_buffer('PSF_fft', PSF_fft.to('cuda'))
        self.padded_matrix = torch.zeros((1, 300, 2304, 2304), dtype=torch.float32,device=self.PSF_fft.device)
        self.relu = nn.ReLU()
        # 将 PSF_fft 移动到模型所在的设备
        #self.PSF_fft = None  # 初始化占位符，稍后将在设备上赋值
        #
        def process_image_tensor(tensor_image, lenses_info):
            """
            处理输入的张量图像，按照 lenses_info 提供的透镜区域信息进行裁剪并堆叠。

            Args:
                tensor_image (torch.Tensor): 输入图像的张量 (C, H, W)，例如 (1, 2304, 2304)。
                lenses_info (list of tuples): 每个透镜区域的 (x, y, w, h) 坐标信息。

            Returns:
                torch.Tensor: 堆叠后的裁剪图像 (N, C, 600, 600)，N 是透镜的数量。
            """

            # 假设 tensor_image 为 3D 张量 (C, H, W)
            C, H, W = tensor_image.shape

            cropped_images = []

            for x, y, w, h in lenses_info:
                # 裁剪透镜区域
                lens_image = tensor_image[:, y:y + h, x:x + w]

                # 检查图像尺寸，必要时进行填充
                if lens_image.shape[1] < 600 or lens_image.shape[2] < 600:
                    new_image = torch.zeros((C, 600, 600), dtype=tensor_image.dtype,
                                            device=tensor_image.device)  # 创建空白图像
                    offset_y = (600 - lens_image.shape[1]) // 2
                    offset_x = (600 - lens_image.shape[2]) // 2
                    new_image[:, offset_y:offset_y + lens_image.shape[1],
                    offset_x:offset_x + lens_image.shape[2]] = lens_image
                    cropped_images.append(new_image)
                else:
                    # 如果已经是 600x600 以上的尺寸，直接使用
                    cropped_images.append(lens_image)

            # 将裁剪后的图像堆叠成一个新的 tensor
            stacked_images = torch.stack(cropped_images, dim=0)  # 堆叠后维度 (N, C, 600, 600)

            return stacked_images

    def consistency(self,matrix1):

        def pad_and_assign(matrix):
            # 只更新张量中的数据，不重新分配内存
            self.padded_matrix[:, :,852:1452, 852:1452].copy_(matrix)
            return self.padded_matrix




        padded_matrix1 = pad_and_assign(matrix1)



        summed_matrix = torch.zeros((2304, 2304), device=padded_matrix1.device)

        # 逐层处理 padded_matrix1 的每一张图像，减少显存占用
        for i in range(padded_matrix1.shape[0]):
            # 对第 i 层图像进行 FFT 操作
            fft_image = torch.fft.fft2(padded_matrix1[0,i])

            # 对第 i 层的 FFT 结果与 PSF_fft[i] 进行乘法，再进行 IFFT 操作
            layer_ifft = torch.real(torch.fft.ifft2(fft_image * self.PSF_fft[i]))

            # 将每层的结果加到 summed_matrix 中
            summed_matrix += layer_ifft


        cropped_matrix = summed_matrix[128:-128, 128:-128]
        cropped_matrix = cropped_matrix.unsqueeze(0).unsqueeze(3)
        cropped_matrix = cropped_matrix / cropped_matrix.max()

        return cropped_matrix

    def down_sampling(self,tensor, target_size=(600, 600)):
        """
        将输入的 4D tensor (batch_size, height, width, channels) 调整为指定的 target_size。

        参数:
        tensor (torch.Tensor): 输入的 tensor，形状为 (batch_size, height, width, channels)
        target_size (tuple): 目标大小，形如 (target_height, target_width)

        返回:
        torch.Tensor: 调整大小后的 tensor，形状为 (batch_size, target_height, target_width, channels)
        """
        # 先调整维度顺序，使其符合 interpolate 的输入要求 (batch_size, channels, height, width)
        #tensor = tensor.permute(0, 3, 1, 2)

        # 使用 interpolate 进行缩放
        resized_tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

        # 调整回原来的维度顺序 (batch_size, height, width, channels)
        resized_tensor = resized_tensor.permute(0, 2, 3, 1)

        return resized_tensor


    def forward(self, x):
        # 初始的下采样处理
        # device = next(self.parameters()).device
        #
        # # 将 PSF_fft 移动到模型设备上
        # self.PSF_fft = self.PSF_fft.to(device)
        x_watch = x[:,26:27,:,:]
        #调整维度，从（1，1，600，600）到（1，600，600，1）
        x_watch = x_watch.permute(0,2,3,1)
        x = self.downsample_head(x)

        # 三阶段处理，每个阶段都共享相同的 restoration_net

        #阶段1
        for i in range(3):

            x = self.restoration_net(x)
            x =self.relu(x)
            x_FP =self.ImageResizer(x)
            #1,2048,2048,1->1,600,600,1
            x_FP=self.down_sampling(x_FP)
            DV =(x_watch)/(x_FP+1e-20)
            BP = x_FP*DV
            #维度调整
            BP = BP.permute(0, 3, 1, 2)
            x = x * BP






        return x

class CombinedModel_deeper_mutistage(nn.Module):
    def __init__(self):
        super(CombinedModel_deeper_mutistage, self).__init__()
        self.downsample_head = DownsampleHead_location()

        img_channel = 300
        width = 64

        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]

        # 初始化 restoration_net（共享权重）
        self.restoration_net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        coordinates_path = '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data_prepare/coordinates.txt'  # 坐标信息文件路径
        #
        # # 生成一个随机的 3D 张量作为输入
        self.relu = nn.ReLU()
        self.lenses_info = []
        with open(coordinates_path, 'r') as file:
            for line in file:
                x, y, w, h = map(int, line.split())
                self.lenses_info.append((x, y, w, h))

        self.lenses_info =torch.tensor(self.lenses_info)

        #获取设备：self.restoration_net
        self.ImageResizer = Forward_Net()
        #self.PSF_fft = PSF_fft

        self.relu = nn.ReLU()
        # 将 PSF_fft 移动到模型所在的设备
        #self.PSF_fft = None  # 初始化占位符，稍后将在设备上赋值
        #

    def fp2stack(self,original_image):

        #去掉0，1维度
        original_image = original_image.squeeze(0)
        original_image = original_image.squeeze(0)
        # 处理每个透镜区域
        cropped_images = []
        for x, y, w, h in self.lenses_info:
            lens_image = original_image[y:y + h, x:x + w]

            # 检查图像尺寸，进行必要的填充
            if lens_image.shape[0] < 600 or lens_image.shape[1] < 600:
                new_image = torch.zeros((600, 600), dtype=torch.float32)  # 创建一个新的空白图像
                offset_y = (600 - lens_image.shape[0]) // 2
                offset_x = (600 - lens_image.shape[1]) // 2
                new_image[offset_y:offset_y + lens_image.shape[0], offset_x:offset_x + lens_image.shape[1]] = lens_image
                cropped_images.append(new_image)
            else:
                cropped_images.append(lens_image)

        # 将所有裁剪后的图像堆叠成一个新的tensor
        stacked_images = torch.stack(cropped_images, dim=0)

        stacked_images =stacked_images.permute(1,2,0)
        stacked_images =stacked_images.unsqueeze(0)

        return stacked_images.cuda()

    def down_sampling(self,tensor, target_size=(600, 600)):
        """
        将输入的 4D tensor (batch_size, height, width, channels) 调整为指定的 target_size。

        参数:
        tensor (torch.Tensor): 输入的 tensor，形状为 (batch_size, height, width, channels)
        target_size (tuple): 目标大小，形如 (target_height, target_width)

        返回:
        torch.Tensor: 调整大小后的 tensor，形状为 (batch_size, target_height, target_width, channels)
        """
        # 先调整维度顺序，使其符合 interpolate 的输入要求 (batch_size, channels, height, width)
        #tensor = tensor.permute(0, 3, 1, 2)

        # 使用 interpolate 进行缩放
        resized_tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

        # 调整回原来的维度顺序 (batch_size, height, width, channels)
        resized_tensor = resized_tensor.permute(0, 2, 3, 1)

        return resized_tensor


    def forward(self, x):
        # 初始的下采样处理
        # device = next(self.parameters()).device
        #
        # # 将 PSF_fft 移动到模型设备上
        # self.PSF_fft = self.PSF_fft.to(device)
        x_input = x[:,0:-1,:,:]  #相机拍到的
        #调整维度，从（1，1，600，600）到（1，600，600，1）
        x_input = x_input.permute(0,2,3,1)
        x_watch = x[:,-2:-1,:,:]
        x_watch = x_watch.permute(0, 2, 3, 1)
        x = self.downsample_head(x)

        # 三阶段处理，每个阶段都共享相同的 restoration_net


        for i in range(2):

            x = self.restoration_net(x)
            x =self.relu(x)
            x_FP =self.ImageResizer(x)

            x_FP_down = self.fp2stack(x_FP) #2048->27
            DV =(x_input)/(x_FP_down+1e-6)
            #tensor concate
            DV_watch = torch.cat((DV, x_watch), 3)
            #维度调整
            DV_watch = DV_watch.permute(0, 3, 1, 2)
            DV_watch_up = self.downsample_head(DV_watch)
            BP =self.restoration_net(DV_watch_up)
            #维度调整
            #BP = BP.permute(0, 3, 1, 2)
            x = x * BP
        return x #, x_FP
#x是重建结果，x_FP是
class DownsampleHead_Forward(nn.Module):
    def __init__(self):
        super(DownsampleHead_Forward, self).__init__()
        self.conv1 = nn.Conv2d(300, 128, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 128, 600, 600)
        self.conv2 = nn.Conv2d(128, 27, kernel_size=3, stride=1, padding=1)  # Output shape: (1, 300, 600, 600)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class CombinedModel_forward(nn.Module):
    def __init__(self):
        super(CombinedModel_forward, self).__init__()
        self.downsample_head = DownsampleHead_Forward()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 27
        width = 32

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        self.relu=nn.ReLU()
    def forward(self, x):


        x = self.downsample_head(x)
        x = self.restoration_net(x)

        x = self.relu(x)
        return x


class CombinedModel_FP(nn.Module):
    def __init__(self):
        super(CombinedModel_FP, self).__init__()
        self.downsample_head = DownsampleHead()
        #self.SEBottleneck = SEBottleneck()


        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        x = self.relu(x)
        return x


class CombinedModel_muti_stage_ImageResizer(nn.Module):
    def __init__(self):
        super(CombinedModel_muti_stage_ImageResizer, self).__init__()
       # self.downsample_head = DownsampleHead()
        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 2, 2, 2,4]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2, 2,2]


        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]




        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        #self.ImageResizer =ImageResizer()
        self.ImageResizer =ImageResizerLow()

    def forward(self, x):
        #x = self.downsample_head(x)
        x = self.restoration_net(x)
        x = self.ImageResizer(x)
        return x




class Forward_Net(nn.Module):
    def __init__(self):
        super(Forward_Net, self).__init__()
       # self.downsample_head = DownsampleHead()
        img_channel = 300
        width = 32

        # enc_blks = [2, 2, 2, 2, 2,4]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2, 2,2]


        enc_blks = [2, 2, 2, 2,4]
        middle_blk_num = 8
        dec_blks = [2, 2, 2, 2,2]



        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        #self.ImageResizer =ImageResizer()
        self.ImageResizer =ImageResizerLow()

    def forward(self, x):
        #x = self.downsample_head(x)
        x = self.restoration_net(x)
        x = self.ImageResizer(x)
        return x



class CombinedModel_forward_model(nn.Module):
    def __init__(self):
        super(CombinedModel_forward_model, self).__init__()
       # self.downsample_head = DownsampleHead()
        img_channel = 300
        width = 64

        enc_blks = [2, 2, 2, 2, 2,4]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2, 2,2]



        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        self.ImageResizer =ImageResizer()

    def forward(self, x):
        #x = self.downsample_head(x)
        x = self.restoration_net(x)
        x = self.ImageResizer(x)
        return x






class CombinedModel_deeper_w32(nn.Module):
    def __init__(self):
        super(CombinedModel_deeper_w32, self).__init__()
        self.downsample_head = DownsampleHead_location()
        # self.downsample_head=DownsampleHead()
        #self.SEBottleneck = SEBottleneck()
        #前值
        # enc_blks = [2, 2, 2, 2, 2,4]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2, 2,2]

        img_channel = 300
        width = 32

        enc_blks = [2, 2, 2, 2, 2,4]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2, 2,2]

        self.restoration_net =NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    def forward(self, x):
        x = self.downsample_head(x)

        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        return x


class MultiScaleHead(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        super(MultiScaleHead, self).__init__()

        # 转置卷积层
        self.deconv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # 假设输入为不同层的输出，逐层上采样
        for in_channels in in_channels_list:
            self.deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.norm_layers.append(nn.BatchNorm2d(out_channels))

        # 最后的卷积层用于将输出调整到目标尺寸
        self.final_conv = nn.Conv2d(len(in_channels_list) * out_channels, 1, kernel_size=1)

        # 目标尺寸
        self.target_height, self.target_width = target_size

    def forward(self, features):
        # 特征列表
        upsampled_features = []

        for i, feature in enumerate(features):
            # 上采样每个特征图
            upsampled = self.deconv_layers[i](feature)
            upsampled = self.norm_layers[i](upsampled)
            upsampled_features.append(upsampled)

        # 将上采样后的特征图拼接在一起
        combined_features = torch.cat(upsampled_features, dim=1)  # 在通道维度拼接

        # 通过最后的卷积层调整输出通道数
        output = self.final_conv(combined_features)

        # 确保输出的尺寸是 (1, 300, 600, 600)
        output = nn.functional.interpolate(output, size=(self.target_height, self.target_width), mode='bilinear',
                                           align_corners=False)

        return output


class SwinTransformer_head(nn.Module):
    def __init__(self):
        super(SwinTransformer_head, self).__init__()
        self.conv1x1 = nn.Conv2d(768, 300, kernel_size=1)
        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1) # 输出大小约为 38x38
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1) # 输出大小约为 76x76
        self.upconv3 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1) # 输出大小约为 152x152
        self.upconv4 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1) # 输出大小约为 304x304
        self.upconv5 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1) # 输出大小约为 608x608
        # 1x1 卷积调整最终尺寸
        self.final_conv = nn.Conv2d(300, 300, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))
        # 调整到600x600
        x = x[:,:,:600,:600]
        x = self.final_conv(x)
        return x


class SwinTransformer_fusion_head(nn.Module):
    def __init__(self):
        super(SwinTransformer_fusion_head, self).__init__()
        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(96, 300, kernel_size=1)  # 输入通道数从 96 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 300, 300, 150x150 -> 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 300, 300, 300x300 -> 600x600

        # 1x1 卷积调整最终尺寸
        self.final_conv = nn.Conv2d(300, 300, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 96, 150, 150] -> 输出 [1, 300, 150, 150]
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]

        # 最终卷积调整
        x = self.final_conv(x)  # [1, 300, 600, 600]

        return x


class SwinTransformerFusionHead_base(nn.Module):
    def __init__(self):
        super(SwinTransformerFusionHead_base, self).__init__()

        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(128, 300, kernel_size=1)  # 输入通道数从 128 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 从 150x150 -> 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 从 300x300 -> 600x600

        # 1x1 卷积调整最终尺寸
        self.final_conv = nn.Conv2d(300, 300, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 128, 150, 150] -> 输出 [1, 300, 150, 150]
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]

        # 最终卷积调整
        x = self.final_conv(x)  # [1, 300, 600, 600]

        return x


class SwinTransformerFusionHead_Large(nn.Module):
    def __init__(self):
        super(SwinTransformerFusionHead_Large, self).__init__()

        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(192, 300, kernel_size=1)  # 输入通道数从 192 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 从 150x150 -> 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 从 300x300 -> 600x600

        # 1x1 卷积调整最终尺寸
        self.final_conv = nn.Conv2d(300, 300, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 192, 150, 150] -> 输出 [1, 300, 150, 150]
        #print(f"After conv1x1: {x.shape}")
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
       # print(f"After upconv1: {x.shape}")
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]
        #print(f"After upconv2: {x.shape}")

        # 最终卷积调整
        x = self.final_conv(x)  # [1, 300, 600, 600]
        #print(f"After final_conv: {x.shape}")

        return x

class Swin_Transformer_Recon(nn.Module):
    def __init__(self):
        super(Swin_Transformer_Recon, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=tuple[2, 2, 6, 2],
        # pretrain_img_size=224,
        # embed_dim=96,
        # window_size=7,
        # ape=False,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)
        x = x[3]
        x = self.bottleneck(x)
        return x

class Swin_Transformer_Recon_fusion(nn.Module):
    def __init__(self):
        super(Swin_Transformer_Recon_fusion, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[16, 16, 48, 16],
        # pretrain_img_size=224,
        # embed_dim=96,
        window_size=20,
        # ape=False,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x





class Swin_Transformer_Recon_fusion_half(nn.Module):
    def __init__(self):
        super(Swin_Transformer_Recon_fusion_half, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[8, 8, 24, 8],
        # pretrain_img_size=224,
        # embed_dim=96,
        window_size=20,
        # ape=False,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x



class Swin_Transformer_Recon_fusion_mini(nn.Module):
    def __init__(self):
        super(Swin_Transformer_Recon_fusion_mini, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[4, 4, 12, 4],
        # pretrain_img_size=224,
        # embed_dim=96,
        window_size=20,
        # ape=False,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x



class Swin_Transformer_tiny(nn.Module):
    def __init__(self):
        super(Swin_Transformer_tiny, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        # pretrain_img_size=224,
        embed_dim=96,
        window_size=7,
        # ape=False,
        drop_path_rate=0.2,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x

class Swin_Transformer_tiny_freeze(nn.Module):
    def __init__(self):
        super(Swin_Transformer_tiny_freeze, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        # pretrain_img_size=224,
        embed_dim=96,
        window_size=7,
        # ape=False,
        drop_path_rate=0.2,
        # patch_norm=True,
        # use_checkpoint=False
    )
        # ====== 关键：冻结 restoration_net 的参数 =======
        for param in self.restoration_net.parameters():
            param.requires_grad = False
        # ============================================
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x


class Swin_Transformer_small(nn.Module):
    def __init__(self):
        super(Swin_Transformer_small, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        # pretrain_img_size=224,
        embed_dim=96,
        window_size=7,
        # ape=False,
        drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x

class Swin_Transformer_base(nn.Module):
    def __init__(self):
        super(Swin_Transformer_base, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformerFusionHead_base()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        # pretrain_img_size=224,
        embed_dim=128,
        window_size=7,
        # ape=False,
        drop_path_rate=0.5,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x

class Swin_Transformer_large(nn.Module):
    def __init__(self):
        super(Swin_Transformer_large, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformerFusionHead_Large()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        # pretrain_img_size=224,
        embed_dim=192,
        window_size=7,
        # ape=False,
        drop_path_rate=0.2,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x







class Swin_Transformer_Recon_fusion_1(nn.Module):
    def __init__(self):
        super(Swin_Transformer_Recon_fusion_1, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        self.bottleneck=SwinTransformer_fusion_head()


        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[16, 16, 48, 16],
        # pretrain_img_size=224,
        # embed_dim=96,
        window_size=20,
        # ape=False,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.bottleneck(x)
        return x


class MAE_Bottleneck(nn.Module):
    def __init__(self):
        super(MAE_Bottleneck, self).__init__()

        # 1. 降维操作
        self.reduce_channels = nn.Linear(in_features=768, out_features=28, bias=False)  # 线性降维 (768 -> 28)

        # 2. 转换为 2D 特征图
        self.to_2d = nn.Unflatten(dim=1, unflattened_size=(64, 64))  # 4096 -> 64x64 (假设输入为 64x64 Patch tokens)

        # 3. 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(28)
        self.relu = nn.ReLU(inplace=True)

        # 4. 转置卷积逐步扩展空间尺寸
        self.transposed_conv1 = nn.ConvTranspose2d(
            in_channels=28,
            out_channels=28,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.transposed_conv2 = nn.ConvTranspose2d(
            in_channels=28,
            out_channels=28,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.transposed_conv3 = nn.ConvTranspose2d(
            in_channels=28,
            out_channels=28,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.transposed_conv4 = nn.ConvTranspose2d(
            in_channels=28,
            out_channels=28,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # 5. 卷积调整到最终形状
        self.adjust_conv = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入 x 的形状: (1, 4096, 768)
        # Step 1: 降维操作
        x = self.reduce_channels(x)  # (1, 4096, 768) -> (1, 4096, 28)

        # Step 2: 转换为 2D 特征图
        x = self.to_2d(x)  # (1, 4096, 28) -> (1, 28, 64, 64)

        # Step 3: 调整为 NCHW 格式并进行 1x1 卷积
        x = x.permute(0, 3, 1, 2).contiguous()  # (1, 64, 64, 28) -> (1, 28, 64, 64)
        x = self.conv1x1(x)  # (1, 28, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)

        # Step 4: 转置卷积逐步扩展尺寸
        x = self.transposed_conv1(x)  # (1, 28, 128, 128)
        x = self.transposed_conv2(x)  # (1, 28, 256, 256)
        x = self.transposed_conv3(x)  # (1, 28, 512, 512)
        x = self.transposed_conv4(x)  # (1, 28, 1024, 1024)

        # Step 5: 卷积调整到目标形状
        x = self.adjust_conv(x)  # (1, 28, 1024, 1024)

        # Step 6: 裁剪到最终形状
        x = x[:, :, :600, :600]  # 确保最终尺寸为 (1, 28, 600, 600)

        return x



#sys.path.append('/home/hefengcs/models/mae-main')
#from models_mae import mae_vit_base_patch16_dec512d8b_2048

# # 假设 MAE_Bottleneck 和 Swin_Transformer_tiny 已经定义好
# class MAE_2048_Swin_Transformer_Tiny(nn.Module):
#     def __init__(self, mae_checkpoint_path="/mnt/raid/MAE/output_dir/checkpoint-400.pth"):
#         super(MAE_2048_Swin_Transformer_Tiny, self).__init__()
#
#         # 1. 加载 MAE encoder
#         self.mae_model = mae_vit_base_patch16_dec512d8b_2048()
#
#         if mae_checkpoint_path is not None:
#             print(f"Loading MAE pre-trained weights from: {mae_checkpoint_path}")
#             checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
#             self.mae_model.load_state_dict(checkpoint['model'], strict=False)
#
#         # 2. 定义 bottleneck
#         self.bottleneck = MAE_Bottleneck()
#
#         # 3. 加载 Swin Transformer Tiny
#         self.swin_transformer = Swin_Transformer_tiny()
#
#     def forward(self, x):
#         # Step 1: 通过 MAE 的 encoder
#         latent, _, _ = self.mae_model.forward_encoder(x, mask_ratio=0)  # 输入形状: (1, 3, 2048, 2048)
#         latent = latent[:, 1:, :]  # 去掉 CLS token
#         # print(f"Latent shape after MAE encoder: {latent.shape}")  # (1, 4096, 768)
#
#         # Step 2: 通过 Bottleneck 进行维度转换
#         latent_reshaped = self.bottleneck(latent)  # 输入形状: (1, 4096, 768)，输出: (1, 28, 600, 600)
#         # print(f"Shape after bottleneck: {latent_reshaped.shape}")
#
#         # Step 3: 传递到 Swin Transformer
#         output = self.swin_transformer(latent_reshaped)  # 输出形状取决于 Swin Transformer 的设计
#         # print(f"Final output shape: {output.shape}")
#
#         return output


# class MAE_2048_Swin_Transformer_Tiny(nn.Module):
#     def __init__(self, mae_checkpoint_path="/mnt/raid/MAE/output_dir/checkpoint-400.pth"):
#         super(MAE_2048_Swin_Transformer_Tiny, self).__init__()
#
#         # 1. 只加载并冻结 MAE Encoder
#         self.mae_encoder = self.load_mae_encoder(mae_checkpoint_path)
#
#         # 2. 定义 Bottleneck，用于维度转换
#         self.bottleneck = MAE_Bottleneck()
#
#         # 3. 加载 Swin Transformer Tiny
#         self.swin_transformer = Swin_Transformer_tiny()
#
#     def load_mae_encoder(self, mae_checkpoint_path):
#         """
#         加载并冻结 MAE 的 Encoder 部分。
#         """
#         print("Loading MAE pre-trained encoder...")
#         mae_model = mae_vit_base_patch16_dec512d8b_2048()
#         checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
#         mae_model.load_state_dict(checkpoint['model'], strict=False)
#
#         # 提取 Encoder 部分
#         mae_encoder = nn.Module()
#         mae_encoder.patch_embed = mae_model.patch_embed
#         mae_encoder.cls_token = mae_model.cls_token
#         mae_encoder.pos_embed = mae_model.pos_embed
#         mae_encoder.blocks = mae_model.blocks
#         mae_encoder.norm = mae_model.norm
#
#         # 冻结 Encoder 的参数
#         for param in mae_encoder.parameters():
#             param.requires_grad = False
#
#         return mae_encoder
#
#     def forward_encoder(self, x):
#         """
#         使用 MAE 的 Encoder 提取特征。
#         """
#         # 1. Patch Embedding
#         x = self.mae_encoder.patch_embed(x)
#         x = x + self.mae_encoder.pos_embed[:, 1:, :]  # 加位置编码
#
#         # 2. 添加 CLS token
#         cls_token = self.mae_encoder.cls_token + self.mae_encoder.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # 3. Encoder blocks
#         for blk in self.mae_encoder.blocks:
#             x = blk(x)
#         x = self.mae_encoder.norm(x)
#
#         return x
#
#     def forward(self, x):
#         # Step 1: 通过 MAE 的 Encoder
#         latent = self.forward_encoder(x)  # 输入形状: (1, 3, 2048, 2048)
#         latent = latent[:, 1:, :]  # 去掉 CLS token
#         # print(f"Latent shape after MAE encoder: {latent.shape}")  # (1, 4096, 768)
#
#         # Step 2: 通过 Bottleneck 进行维度转换
#         latent_reshaped = self.bottleneck(latent)  # 输入: (1, 4096, 768)，输出: (1, 28, 600, 600)
#         # print(f"Shape after bottleneck: {latent_reshaped.shape}")
#
#         # Step 3: 传递到 Swin Transformer
#         output = self.swin_transformer(latent_reshaped)  # 输出形状取决于 Swin Transformer 的设计
#         # print(f"Final output shape: {output.shape}")
#
#         return output


class UpsampleHead(nn.Module):
    def __init__(self, input_dim=768, target_channels=300, target_size=(600, 600)):
        super(UpsampleHead, self).__init__()
        self.target_size = target_size

        # 将 lambda 替换为一个方法
        self.reshape = self._reshape

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.channel_reduce = nn.Conv2d(input_dim, target_channels, kernel_size=1)

    def _reshape(self, x):
        """重构 Patch Token 到特征图"""
        return x.view(-1, 768, 20, 20)

    def crop(self, x, target_size):
        _, _, h, w = x.size()
        target_h, target_w = target_size
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return x[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

    def forward(self, x):
        x = self.reshape(x)  # 调用方法代替 lambda
        x = self.upsample(x)
        x = self.crop(x, self.target_size)
        x = self.channel_reduce(x)
        return x


# class MAE_2048_simple_head(nn.Module):
#     def __init__(self, mae_checkpoint_path="/mnt/raid/MAE/output_dir/checkpoint-400.pth",pretrained=False):
#         super(MAE_2048_simple_head, self).__init__()
#         self.pretrained = pretrained
#         # 1. 只加载并冻结 MAE Encoder
#         self.mae_encoder = self.load_mae_encoder(mae_checkpoint_path)
#
#         self.UpsampleHead =UpsampleHead(input_dim=768, target_channels=300, target_size=(600, 600))
#         # 2. 定义 Bottleneck，用于维度转换
#         # self.bottleneck = MAE_Bottleneck()
#         #
#         # # 3. 加载 Swin Transformer Tiny
#         # self.swin_transformer = Swin_Transformer_tiny()
#
#     def load_mae_encoder(self, mae_checkpoint_path):
#         """
#         加载并冻结 MAE 的 Encoder 部分。
#         """
#         print("Loading MAE pre-trained encoder...")
#         mae_model = mae_vit_base_patch16_dec512d8b_2048()
#         if self.pretrained == True:
#             checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
#             mae_model.load_state_dict(checkpoint['model'], strict=False)
#
#         # 提取 Encoder 部分
#         mae_encoder = nn.Module()
#         mae_encoder.patch_embed = mae_model.patch_embed
#         mae_encoder.cls_token = mae_model.cls_token
#         mae_encoder.pos_embed = mae_model.pos_embed
#         mae_encoder.blocks = mae_model.blocks
#         mae_encoder.norm = mae_model.norm
#
#         # 冻结 Encoder 的参数
#         if self.pretrained == True:
#             for param in mae_encoder.parameters():
#                 param.requires_grad = False
#
#         return mae_encoder
#
#     def forward_encoder(self, x):
#         """
#         使用 MAE 的 Encoder 提取特征。
#         """
#         # 1. Patch Embedding
#         x = self.mae_encoder.patch_embed(x)
#         x = x + self.mae_encoder.pos_embed[:, 1:, :]  # 加位置编码
#
#         # 2. 添加 CLS token
#         cls_token = self.mae_encoder.cls_token + self.mae_encoder.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # 3. Encoder blocks
#         for blk in self.mae_encoder.blocks:
#             x = blk(x)
#         x = self.mae_encoder.norm(x)
#
#         return x
#
#     def forward(self, x):
#         # Step 1: 通过 MAE 的 Encoder
#         latent = self.forward_encoder(x)  # 输入形状: (1, 3, 2048, 2048)
#         latent = latent[:, 1:, :]  # 去掉 CLS token
#         # print(f"Latent shape after MAE encoder: {latent.shape}")  # (1, 4096, 768)
#
#         output = self.UpsampleHead(latent)
#
#         return output
class UpsampleHead_600(nn.Module):
    def __init__(self, input_dim=768, target_channels=300, target_size=(600, 600)):
        super(UpsampleHead_600, self).__init__()
        self.target_size = target_size

        # 重构 Patch Token 到特征图: 20x20
        self.reshape = self._reshape

        # 上采样：将特征图从 20x20 放大到 600x600
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=4, stride=2, padding=1),  # 20x20 -> 40x40
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=4, stride=2, padding=1),  # 40x40 -> 80x80
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=4, stride=2, padding=1),  # 80x80 -> 160x160
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim // 8, input_dim // 16, kernel_size=4, stride=2, padding=1),  # 160x160 -> 320x320
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim // 16, input_dim // 32, kernel_size=4, stride=2, padding=1),  # 320x320 -> 600x600
            nn.ReLU()
        )

        # 通道压缩
        self.channel_reduce = nn.Conv2d(input_dim // 32, target_channels, kernel_size=1)

    def _reshape(self, x):
        """重构 Patch Token 到特征图"""
        batch_size, num_patches, embed_dim = x.size()
        h = w = int(num_patches ** 0.5)  # 假设是方形特征图
        return x.view(batch_size, embed_dim, h, w)

    def forward(self, x):
        x = self.reshape(x)  # 重构为特征图 (B, 768, 20, 20)
        x = self.upsample(x)  # 上采样到 (B, C, 600, 600)
        x = self.channel_reduce(x)  # 压缩通道数到 300
        x = x[:,:,:600,:600]  # 裁剪到 600x600
        return x

#from models_mae import mae_vit_base_patch16_dec512d8b_600

# class MAE_600_simple_head(nn.Module):
#     def __init__(self, mae_checkpoint_path="/mnt/raid/MAE600/output_dir/checkpoint-540.pth",pretrained=False):
#         super(MAE_600_simple_head, self).__init__()
#         self.pretrained = pretrained
#         # 1. 只加载并冻结 MAE Encoder
#         self.mae_encoder = self.load_mae_encoder(mae_checkpoint_path)
#
#         self.UpsampleHead =UpsampleHead_600()
#         # 2. 定义 Bottleneck，用于维度转换
#         # self.bottleneck = MAE_Bottleneck()
#         #
#         # # 3. 加载 Swin Transformer Tiny
#         # self.swin_transformer = Swin_Transformer_tiny()
#
#     def load_mae_encoder(self, mae_checkpoint_path):
#         """
#         加载并冻结 MAE 的 Encoder 部分。
#         """
#
#         mae_model = mae_vit_base_patch16_dec512d8b_600()
#
#         if self.pretrained == True:
#             print("Loading MAE pre-trained encoder...")
#             checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
#             mae_model.load_state_dict(checkpoint['model'], strict=False)
#         else:
#             print("Train from the scratch...")
#
#         # 提取 Encoder 部分
#         mae_encoder = nn.Module()
#         mae_encoder.patch_embed = mae_model.patch_embed
#         mae_encoder.cls_token = mae_model.cls_token
#         mae_encoder.pos_embed = mae_model.pos_embed
#         mae_encoder.blocks = mae_model.blocks
#         mae_encoder.norm = mae_model.norm
#
#         # 冻结 Encoder 的参数
#         # if self.pretrained == True:
#         #     for param in mae_encoder.parameters():
#         #         param.requires_grad = False
#
#         return mae_encoder
#
#     def forward_encoder(self, x):
#         """
#         使用 MAE 的 Encoder 提取特征。
#         """
#         # 1. Patch Embedding
#         x = self.mae_encoder.patch_embed(x)
#         x = x + self.mae_encoder.pos_embed[:, 1:, :]  # 加位置编码
#
#         # 2. 添加 CLS token
#         cls_token = self.mae_encoder.cls_token + self.mae_encoder.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # 3. Encoder blocks
#         for blk in self.mae_encoder.blocks:
#             x = blk(x)
#         x = self.mae_encoder.norm(x)
#
#         return x
#
#     def forward(self, x):
#         # Step 1: 通过 MAE 的 Encoder
#         latent = self.forward_encoder(x)  # 输入形状: (1, 3, 2048, 2048)
#         latent = latent[:, 1:, :]  # 去掉 CLS token
#         # print(f"Latent shape after MAE encoder: {latent.shape}")  # (1, 4096, 768)
#
#         output = self.UpsampleHead(latent)
#
#         return output
class Recon_Head(nn.Module):
    def __init__(self):
        super(Recon_Head, self).__init__()
        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(96, 300, kernel_size=1)  # 输入通道数从 96 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 600x600

        # 1x1 卷积调整最终尺寸，输出通道数变为 28
        self.final_conv = nn.Conv2d(300, 28, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 96, 150, 150] -> 输出 [1, 300, 150, 150]
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]

        # 最终卷积调整通道数到 28
        x = self.final_conv(x)  # [1, 300, 600, 600] -> 输出 [1, 28, 600, 600]

        return x
class Recon_Head_base(nn.Module):
    def __init__(self):
        super(Recon_Head_base, self).__init__()
        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(128, 300, kernel_size=1)  # 输入通道数从 96 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 600x600

        # 1x1 卷积调整最终尺寸，输出通道数变为 28
        self.final_conv = nn.Conv2d(300, 28, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 96, 150, 150] -> 输出 [1, 300, 150, 150]
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]

        # 最终卷积调整通道数到 28
        x = self.final_conv(x)  # [1, 300, 600, 600] -> 输出 [1, 28, 600, 600]

        return x



class Recon_Head_large(nn.Module):
    def __init__(self):
        super(Recon_Head_large, self).__init__()
        # 1x1 卷积，调整通道数
        self.conv1x1 = nn.Conv2d(192, 300, kernel_size=1)  # 输入通道数从 96 开始

        # 转置卷积层序列
        self.upconv1 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 300x300
        self.upconv2 = nn.ConvTranspose2d(300, 300, kernel_size=4, stride=2, padding=1)  # 输出大小约为 600x600

        # 1x1 卷积调整最终尺寸，输出通道数变为 28
        self.final_conv = nn.Conv2d(300, 28, kernel_size=1)

    def forward(self, x):
        # 调整通道数
        x = self.conv1x1(x)  # 输入 [1, 96, 150, 150] -> 输出 [1, 300, 150, 150]
        x = F.relu(self.upconv1(x))  # 上采样 [1, 300, 150, 150] -> 输出 [1, 300, 300, 300]
        x = F.relu(self.upconv2(x))  # 上采样 [1, 300, 300, 300] -> 输出 [1, 300, 600, 600]

        # 最终卷积调整通道数到 28
        x = self.final_conv(x)  # [1, 300, 600, 600] -> 输出 [1, 28, 600, 600]

        return x
class Swin_Transformer_tiny_mask(nn.Module):
    def __init__(self):
        super(Swin_Transformer_tiny_mask, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        self.mask_head = Recon_Head()

        self.restoration_net =SwinTransformer(
        in_chans=28,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        # pretrain_img_size=224,
        embed_dim=96,
        window_size=7,
        # ape=False,
        drop_path_rate=0.2,
        # patch_norm=True,
        # use_checkpoint=False
    )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.mask_head(x)
        return x



class Swin_Transformer_small_mask(nn.Module):
    def __init__(self):
        super(Swin_Transformer_small_mask, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        self.mask_head = Recon_Head()

        self.restoration_net = SwinTransformer(
            in_chans=28,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            # pretrain_img_size=224,
            embed_dim=96,
            window_size=7,
            # ape=False,
            drop_path_rate=0.3,
            # patch_norm=True,
            # use_checkpoint=False
        )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.mask_head(x)
        return x


class Swin_Transformer_base_mask(nn.Module):
    def __init__(self):
        super(Swin_Transformer_base_mask, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        self.mask_head = Recon_Head_base()

        self.restoration_net = SwinTransformer(
            in_chans=28,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            # pretrain_img_size=224,
            embed_dim=128,
            window_size=7,
            # ape=False,
            drop_path_rate=0.5,
            # patch_norm=True,
            # use_checkpoint=False
        )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.mask_head(x)
        return x

class Swin_Transformer_large_mask(nn.Module):
    def __init__(self):
        super(Swin_Transformer_large_mask, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        self.mask_head = Recon_Head_large()

        self.restoration_net = SwinTransformer(
            in_chans=28,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            # pretrain_img_size=224,
            embed_dim=192,
            window_size=7,
            # ape=False,
            drop_path_rate=0.2,
            # patch_norm=True,
            # use_checkpoint=False
        )

    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.restoration_net(x)

        x = self.mask_head(x)
        return x
#--------------------------------------对比实验
#Unet
from model.unet.unet_model import UNet
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        self.model = UNet(n_channels=28, n_classes=300, bilinear=True)
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.model(x)


        return x


#VCD
from model.model import UNet_VCD,UNet_Transpose,UNet_Deeper
class VCD_Net(nn.Module):
    def __init__(self):
        super(VCD_Net, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()
        lf_extra = 28  # Number of input channels (example)
        n_slices = 300  # Number of output slices
        output_size = (600, 600)  # Output size


        self.model = UNet_Deeper(lf_extra, n_slices, output_size)
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.model(x)


        return x

#Vit系列
from model.VIT import VIT_base

class Vit_base(nn.Module):
    def __init__(self):
        super(Vit_base, self).__init__()
        #self.downsample_head = DownsampleHead_location()
        #self.bottleneck=SwinTransformer_fusion_head()

        self.model = VIT_base()
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.model(x)


        return x




#RCAN
from model.RCAN import RCAN_Net
class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()



        self.model = RCAN_Net(input_channels=28, output_channels=300, num_features=64, num_rg=5, num_rcab=10, reduction=16)
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.model(x)


        return x

#convnext
from model.convnext import convnext_tiny_with_custom_head
class convnext_tiny(nn.Module):
    def __init__(self):
        super(convnext_tiny, self).__init__()



        self.model = convnext_tiny_with_custom_head()
    def forward(self, x):


        #x =self.SEBottleneck(x)
        x = self.model(x)


        return x

import timm


class ViT_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300, embed_dim=192):
        super().__init__()

        # ViT-Tiny 作为 Encoder
        self.encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
            features_only=True
        )

        # Decoder: 从 (192, 37, 37) 上采样到 (300, 600, 600)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (512, 74, 74)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256, 148, 148)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 296, 296)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # (64, 592, 592)
            nn.ReLU(),
            nn.Conv2d(64, out_chans, kernel_size=3, padding=1),  # (300, 600, 600)
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)  # 提取 ViT 特征
        x = features[-1]  # 取最后一层特征 (batch, 192, 37, 37)
        x = self.decoder(x)  # 通过 CNN Decoder 逐步恢复

        # **使用 `F.interpolate()` 让最终尺寸变为 600x600**
        x = F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)
        return x

# 测试模型

class ConvNeXt_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300):
        super().__init__()

        # 加载 ConvNeXt-Tiny
        self.encoder = timm.create_model('convnext_tiny', pretrained=False)
        self.encoder.head = nn.Identity()  # 移除分类头

        # **修改输入通道数**
        original_patch_embed = self.encoder.stem[0]  # 获取原始 Patch Embedding 层
        self.encoder.stem[0] = nn.Conv2d(
            in_chans, 96, kernel_size=4, stride=4  # 适配 28 通道输入
        )

        # Decoder: 从 (batch, 768, 19, 19) 上采样到 (batch, 300, 600, 600)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (512, 38, 38)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256, 76, 76)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 152, 152)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # (64, 304, 304)
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1),  # (300, 608, 608)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)  # 提取 ConvNeXt 特征
        x = self.decoder(x)  # 通过 CNN Decoder 逐步恢复

        # **最终 resize 到 600x600**
        x = nn.functional.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)
        return x

#PVT model
class PVT_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300, model_name='pvt_v2_b2'):
        super().__init__()

        # 1️⃣ 加载 `PVT` 作为 Encoder
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(3,),  # 取最后一层特征
            in_chans=in_chans
        )

        # 2️⃣ Decoder: **增加 `ConvTranspose2d` 层，确保最终输出 `608x608`**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=1),  # (512, 38, 38)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1),  # (256, 76, 76)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),  # (128, 152, 152)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),   # (64, 304, 304)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),    # (32, 608, 608)
            nn.ReLU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),  # (300, 608, 608)
            nn.Tanh()
        )

    def forward(self, x):
        #print(f"Input Shape: {x.shape}")  # Debug: 查看输入形状

        features = self.encoder(x)  # 提取 PVT 特征
        x = features[-1]  # 取最后一层特征
        #print(f"Encoder Output Shape: {x.shape}")  # Debug: PVT 输出形状

        x = self.decoder(x)  # 通过 CNN Decoder 逐步恢复
        #print(f"Decoder Output Shape: {x.shape}")  # Debug: Decoder 输出形状

        # **裁剪到 (1, 300, 600, 600)**
        x = x[:, :, 4:604, 4:604]  # 取中心区域
        #print(f"Final Output Shape: {x.shape}")  # Debug: 最终形状

        return x


class EfficientNet_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300, model_name='efficientnet_b2'):
        super().__init__()

        # 1️⃣ 加载 `EfficientNet` 作为 Encoder
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(4,),  # 取最后一层特征
            in_chans=in_chans
        )

        # EfficientNet 最后一层输出通道数
        encoder_out_channels = self.encoder.feature_info.get_dicts()[-1]['num_chs']


        # 2️⃣ Decoder: **使用 `ConvTranspose2d` 逐步恢复图像大小**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 512, kernel_size=4, stride=2, padding=1, output_padding=1),  # (512, 38, 38)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1),  # (256, 76, 76)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),  # (128, 152, 152)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),   # (64, 304, 304)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),    # (32, 608, 608)
            nn.ReLU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),  # (300, 608, 608)
            nn.Tanh()
        )

    def forward(self, x):
        # 提取 EfficientNet 特征
        features = self.encoder(x)
        x = features[-1]  # 取最后一层特征

        # 通过 CNN Decoder 逐步恢复
        x = self.decoder(x)

        # **裁剪到 (1, 300, 600, 600)**
        x = x[:, :, 4:604, 4:604]  # 取中心区域

        return x



class SwinIR_Timm(nn.Module):
    def __init__(self, target_size=600, in_chans=28, out_chans=300):
        super().__init__()
        self.target_size = target_size  # 最终输出目标尺寸 600×600
        # 计算最接近 target_size 且为32的倍数的尺寸，供 encoder 使用
        self.input_size = math.ceil(self.target_size / 32) * 32  # 对600而言，608

        # 加载 Swin Transformer Tiny 作为 Encoder，指定 img_size 为 self.input_size（608）
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
            img_size=self.input_size  # 这里设为608
        )
        # 覆盖 patch embedding 模块中的 img_size，避免内部断言错误
        self.encoder.patch_embed.img_size = (self.input_size, self.input_size)

        # 对于 Swin Tiny，features[2] 的输出通道数为 384（非 512），
        # 使用 1×1 卷积将 384 降维到 192，以适配 decoder 的输入
        self.reduce = nn.Conv2d(384, 192, kernel_size=1)

        # 反卷积逐步上采样设计：
        # 假设 encoder 输出特征图尺寸为 38×38（608/16=38），则反卷积层依次上采样：
        # 38 -> 75 -> 150 -> 300 -> 600（最后一层卷积保持尺寸）
        self.decoder = nn.Sequential(
            # 第一层：38 -> 75
            nn.ConvTranspose2d(192, 512, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # 第二层：75 -> 150
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第三层：150 -> 300
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第四层：300 -> 600
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 最后一层：卷积调整通道数，不改变空间尺寸
            nn.Conv2d(64, out_chans, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x 原始尺寸: [N, in_chans, target_size, target_size] 即 600×600
        orig_H, orig_W = x.shape[-2:]
        # 若输入尺寸不等于 encoder 预期尺寸，则调整到 self.input_size（608×608）
        if (orig_H, orig_W) != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # 传入 encoder 得到多尺度特征（列表，形状为 [N, H, W, C]）
        features = self.encoder(x)
        # 选择 features[2]，其形状预期为 [N, H, W, 384]，此时 H = W = 608/16 = 38
        x = features[2]
        # 将特征从 [N, H, W, C] 转换为 [N, C, H, W]
        x = x.permute(0, 3, 1, 2)
        # 如有需要，将特征调整到 38×38（一般不会发生）
        if x.shape[-2:] != (38, 38):
            x = F.interpolate(x, size=(38, 38), mode='bilinear', align_corners=False)
        # 1×1 降维：384 -> 192
        x = self.reduce(x)
        # 通过 decoder 进行反卷积上采样，理论输出尺寸为 600×600
        x = self.decoder(x)
        # 最后确保输出尺寸为 target_size（600×600），以防微小尺寸偏差
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return x

class ResNet50_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300, model_name='resnet50'):
        super().__init__()
        self.img_size = img_size

        # Encoder: 使用 ResNet50，features_only=True，取最后一层特征（layer4）
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(3,),  # 对于 resnet50，layer4 输出
            in_chans=in_chans
        )

        # 用 dummy 输入获得 encoder 实际输出尺寸，注意取最后一个特征
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            feat = self.encoder(dummy)[-1]  # 取列表最后一项
        _, encoder_out_channels, h, w = feat.shape
        #print(f"[Init] Encoder output spatial size: {h}x{w} (channels: {encoder_out_channels})")

        # Decoder: 设计 5 层上采样，将 (B, encoder_out_channels, H, W) → (B, out_chans, ~608, ~608)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 1024, kernel_size=4, stride=2, padding=1),  # H -> 2*H
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 2*H -> 4*H
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4*H -> 8*H
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8*H -> 16*H
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16*H -> 32*H
            nn.ReLU(),
            nn.Conv2d(64, out_chans, kernel_size=3, padding=1),  # 输出 (B, out_chans, S, S)
            nn.Tanh()
        )

    def forward(self, x):
        # x: (B, in_chans, img_size, img_size)
        features = self.encoder(x)  # 返回一个列表
        x = features[-1]  # 取最后一层特征
        x = self.decoder(x)
        # 裁剪中心区域得到精确的 (img_size, img_size)
        S = x.shape[-1]
        start = (S - self.img_size) // 2
        x = x[:, :, start:start + self.img_size, start:start + self.img_size]
        return x


class ResNet101_ImageRestoration(nn.Module):
    def __init__(self, img_size=600, in_chans=28, out_chans=300, model_name='resnet101'):
        super().__init__()
        self.img_size = img_size

        # Encoder: 使用 ResNet101，features_only=True，取最后一层特征（layer4）
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(3,),  # 对于 resnet101，layer4 输出
            in_chans=in_chans
        )

        # 用 dummy 输入获得 encoder 实际输出尺寸，注意取最后一个特征
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            feat = self.encoder(dummy)[-1]  # 取列表最后一项
        _, encoder_out_channels, h, w = feat.shape
        #print(f"[Init] Encoder output spatial size: {h}x{w} (channels: {encoder_out_channels})")

        # Decoder: 设计 5 层上采样，将 (B, encoder_out_channels, H, W) → (B, out_chans, ~608, ~608)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 1024, kernel_size=4, stride=2, padding=1),  # H -> 2*H
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 2*H -> 4*H
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4*H -> 8*H
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8*H -> 16*H
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16*H -> 32*H
            nn.ReLU(),
            nn.Conv2d(64, out_chans, kernel_size=3, padding=1),  # 输出 (B, out_chans, S, S)
            nn.Tanh()
        )

    def forward(self, x):
        # x: (B, in_chans, img_size, img_size)
        features = self.encoder(x)  # 返回一个列表
        x = features[-1]  # 取最后一层特征
        x = self.decoder(x)
        # 裁剪中心区域得到精确的 (img_size, img_size)
        S = x.shape[-1]
        start = (S - self.img_size) // 2
        x = x[:, :, start:start + self.img_size, start:start + self.img_size]
        return x



#MAE补充实验
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import timm
import math

# --------------------------- Config -------------------------------------------
ckpt_path = None # <- change this
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------- Encoder Loader ----------------------------------

def load_mae_encoder(model_name: str = 'vit_small_patch16_224.mae', ckpt: str = None) -> nn.Module:
    model = timm.create_model(model_name, pretrained=(ckpt is None), num_classes=0)
    if ckpt:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
    if hasattr(model, 'cls_token'):
        model.cls_token = None
    return model

# --------------------------- Bottleneck --------------------------------------

class MAEBottleneck(nn.Module):
    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.proj = nn.Linear(embed_dim, num_views * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(num_views, 128, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, num_views, 4, 2, 1),
        )
    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        Bv, N, C = patch_tokens.shape
        x = self.proj(patch_tokens)
        x = x.transpose(1, 2).contiguous()
        side = int(math.sqrt(N))
        x = x.view(Bv, self.num_views, 64, side, side)
        x = x.mean(dim=0)
        return self.deconv(x)

# --------------------------- 3D Head -----------------------------------------

class UpsampleHead600(nn.Module):
    def __init__(self, out_depth: int = 300):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 1, 3, padding=1),
        )
        self.out_depth = out_depth
    def forward(self, x):
        vol = self.conv3d(x)
        vol = F.interpolate(vol, size=(self.out_depth, x.size(3), x.size(4)), mode='trilinear', align_corners=False)
        return vol.squeeze(1)

# --------------------------- MAE-XLFM Wrapper --------------------------------

class MAEXLFM(nn.Module):
    def __init__(self, num_views=28, out_depth=300, model_name='vit_small_patch16_224.mae', ckpt=None):
        super().__init__()
        self.encoder = load_mae_encoder(model_name, ckpt)
        embed_dim = getattr(self.encoder, 'embed_dim', None) or getattr(self.encoder, 'num_features', None)
        self.bottleneck = MAEBottleneck(num_views, embed_dim)
        self.head = UpsampleHead600(out_depth)
        self.num_views = num_views

    @torch.no_grad()
    def _prep_views(self, views: torch.Tensor) -> torch.Tensor:
        B, V, H, W = views.shape
        v = views.view(B * V, 1, H, W)
        return resize(v, [224, 224])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, H, W = x.shape
        assert V == self.num_views
        x_views = self._prep_views(x)
        patch_tokens = self.encoder.forward_features(x_views)
        view_tensor = self.bottleneck(patch_tokens)
        view_tensor = view_tensor.unsqueeze(0).repeat(B, 1, 1, 1)
        vol_in = view_tensor.unsqueeze(1)
        return self.head(vol_in)

# --------------------------- Run ---------------------------------------------

if __name__ == '__main__':
    model = MAEXLFM(ckpt=ckpt_path).to(device).eval()
    x = torch.randn(1, 28, 600, 600).to(device)
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape, "→ output:", y.shape)