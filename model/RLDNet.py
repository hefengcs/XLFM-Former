import os
import argparse
import torch
import numpy as np
import tifffile as tiff
import h5py
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from tqdm import tqdm
import torch
import numpy as np

class RichardsonLucySubModule(torch.nn.Module):
    def __init__(self, init, PSF, device_id):
        super().__init__()
        self.init = init.to(device_id).to(torch.float16)  # 将初始化图像转换为 float16
        self.device_id = device_id

        NxyExt = 128
        Nxy = 2304
        Nz = 300

        # 定义可学习的卷积核参数，并转换为半精度 float16
        self.conv_real = nn.Parameter(torch.rand(1, 1, Nxy, Nxy, dtype=torch.float16) * 0.001)
        self.conv_imag = nn.Parameter(torch.rand(1, 1, Nxy, Nxy, dtype=torch.float16) * 0.001)

    def forward(self, imstack):
        ItN = 60
        BkgMean = 110
        ROISize = 300
        SNR = 200
        NxyExt = 128
        Nxy = 2304
        Nz = 300

        # 使用 float16 进行中间张量计算以减少显存占用
        gpuObjReconTmp = torch.zeros((1, 1, Nxy, Nxy), device=self.device_id, dtype=torch.float16)
        ImgEst = torch.zeros((1, 1, Nxy, Nxy), device=self.device_id, dtype=torch.float32)  # 保持累加张量为 float32
        Ratio = torch.ones((1, 1, Nxy, Nxy), device=self.device_id, dtype=torch.float32)  # 使用 float32 进行比值计算
        Img = imstack.to(self.device_id).to(torch.float16)  # 将输入图像转换为 float16
        ImgMultiView = Img - BkgMean
        ImgMultiView = torch.clamp(ImgMultiView, min=0)
        ImgExp = torch.nn.functional.pad(ImgMultiView, (NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

        gpuObjRecon = self.init.clone()

        for ii in range(ItN):
            ImgEst.fill_(0)
            for jj in range(Nz):
                gpuObjReconTmp[:, :, Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                               Nxy // 2 - ROISize:Nxy // 2 + ROISize] = gpuObjRecon[:, jj, :, :].to(torch.float16)

                # 可学习卷积核（float16）
                conv_kernel = torch.complex(self.conv_real, self.conv_imag)

                # 使用 float32 进行 FFT 和 iFFT
                ImgEst_slice_update = torch.real(
                    torch.fft.ifft2(torch.fft.fft2(gpuObjReconTmp.to(torch.float32)) * conv_kernel.to(torch.float32))
                ) / torch.sum(self.conv_real)

                # 将结果转换为 float32 进行累加
                ImgEst += torch.max(ImgEst_slice_update.to(torch.float32), torch.tensor(0.0, device=ImgEst_slice_update.device))

            ImgExpEst = ImgExp.to(torch.float32)

            Tmp = torch.median(ImgEst)

            Ratio.fill_(1)
            Ratio[:, :, NxyExt:-NxyExt, NxyExt:-NxyExt] = ImgExpEst[:, :, NxyExt:-NxyExt, NxyExt:-NxyExt] / (
                ImgEst[:, :, NxyExt:-NxyExt, NxyExt:-NxyExt] + Tmp / SNR)

            for jj in range(Nz):
                fft_Ratio = torch.fft.fft2(Ratio.to(torch.float32))

                ifft_result = torch.fft.ifft2(fft_Ratio * torch.conj(conv_kernel.to(torch.float32)))

                gpuTmp = torch.real(ifft_result).to(torch.float16) / torch.sum(self.conv_real)

                gpuTmp = torch.maximum(gpuTmp, torch.tensor(0.0, device=self.device_id, dtype=torch.float16))

                gpuObjRecon[:, jj:jj + 1, :, :] *= gpuTmp[:, :, Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                                                         Nxy // 2 - ROISize:Nxy // 2 + ROISize]

        ObjRecon = gpuObjRecon.cpu().numpy().astype(np.float32)
        return ObjRecon



def main():
    # Device configuration
    device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random tensors for init, PSF, and imstack
    init = torch.rand(1,300, 600, 600, device=device_id)  # Initial volume (Nz, Nxy, Nxy)
    PSF = torch.rand(1,300, 2048, 2048, device=device_id)  # Point spread function (Nz, Nxy, Nxy)
    imstack = torch.rand(1,1, 2048, 2048, device=device_id)  # Input image stack (Nz, Nxy, Nxy)

    # Instantiate the RichardsonLucySubModule
    rl_module = RichardsonLucySubModule(init, PSF, device_id)
    rl_module = rl_module.to(device_id)
    # Perform forward pass and get the output
    output = rl_module(imstack)

    # Print the output shape to verify
    print("Output shape:", output.shape)


# Run the main function
if __name__ == "__main__":
    main()