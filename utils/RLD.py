import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import numpy as np
import tifffile as tiff
import h5py
import time
from torch.fft import fft2, ifft2
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler

scaler = GradScaler()

def fftshift(x):
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, dims=(-2, -1))

def ifftshift(x):
    shift0 = x.shape[0] // 2
    shift1 = x.shape[1] // 2
    return torch.roll(x, shifts=(shift0, shift1), dims=(0, 1))

def reConstruct(imstack, init, PSF):
    device = imstack.device
    ItN = 60  # Number of iterations
    BkgMean = 120  # Background noise mean level
    ROISize = 300  # Region of interest size
    SNR = 200  # Signal-to-noise ratio
    NxyExt = 128  # Image extension size
    Nxy = PSF.shape[1] + NxyExt * 2  # Extended image size
    Nz = 300  # Z-axis height

    BkgFilterCoef = 0.0
    x = torch.arange(-Nxy//2, Nxy//2, device=device).float()
    y = torch.arange(-Nxy//2, Nxy//2, device=device).float()
    x, y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(x**2 + y**2)
    Rlimit = 20
    RWidth = 20
    BkgFilter = (torch.cos((R-Rlimit)/RWidth*np.pi)/2+0.5)*(R>=Rlimit)*(R<=(Rlimit+RWidth)) + (R<Rlimit)
    BkgFilter = fftshift(BkgFilter)

    PSF = torch.nn.functional.pad(PSF, (0, 0, NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

    gpuObjReconTmp = torch.zeros((Nxy, Nxy), device=device, dtype=torch.float32)
    ImgEst = torch.zeros((Nxy, Nxy), device=device, dtype=torch.float32)
    Ratio = torch.ones((Nxy, Nxy), device=device, dtype=torch.float32)
    ImgMultiView = torch.clamp(imstack - BkgMean, min=0)
    ImgExp = torch.nn.functional.pad(ImgMultiView, (NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

    gpuObjRecon = init.clone()

    fft_PSF_cache = {}
    for jj in range(Nz):
        PSF_jj = PSF[:, :, jj]
        if jj not in fft_PSF_cache:
            fft_PSF_cache[jj] = fft2(ifftshift(PSF_jj))

    for ii in range(ItN):
        ImgEst.zero_()
        with autocast():  # 开启混合精度计算
            for jj in range(Nz):
                gpuObjReconTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize, Nxy // 2 - ROISize:Nxy // 2 + ROISize] = gpuObjRecon[jj, :, :]

                fft_PSF = fft_PSF_cache[jj]

                ImgEst_slice_update = torch.real(ifft2(fft2(gpuObjReconTmp) * fft_PSF)) / torch.sum(PSF[:, :, jj])
                ImgEst += torch.clamp(ImgEst_slice_update, min=0)

            BkgEst = torch.real(ifft2(fft2(ImgExp - ImgEst) * BkgFilter)) * BkgFilterCoef
            ImgExpEst = ImgExp - BkgEst

            Tmp = torch.median(ImgEst)

            Ratio.fill_(1)
            Ratio[NxyExt:-NxyExt, NxyExt:-NxyExt] = ImgExpEst[NxyExt:-NxyExt, NxyExt:-NxyExt] / (
                        ImgEst[NxyExt:-NxyExt, NxyExt:-NxyExt] + Tmp / SNR)

            for jj in range(Nz):
                fft_PSF_conj = torch.conj(fft_PSF_cache[jj])

                fft_Ratio = fft2(Ratio)
                gpuTmp = torch.real(ifft2(fft_Ratio * fft_PSF_conj)) / torch.sum(PSF[:, :, jj])
                gpuTmp = torch.clamp(gpuTmp, min=0)

                gpuObjRecon[jj, :, :] *= gpuTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                                                Nxy // 2 - ROISize:Nxy // 2 + ROISize]

    return gpuObjRecon.cpu().numpy().astype(np.float32)

if __name__ == "__main__":
    imstack = tiff.imread('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/2048input/00000004.tif').astype(np.float32)
    init = np.ones((300, 600, 600), dtype=np.float32)

    with h5py.File('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/PSF_G.mat', 'r') as f:
        PSF = f['PSF_1'][:]

    PSF = PSF.astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    PSF = np.transpose(PSF, (2, 1, 0))  # Ensure PSF has shape (300, 2048, 2048)
    PSF = torch.from_numpy(PSF).to(device).float()
    imstack = torch.from_numpy(imstack).to(device).float()
    init = torch.from_numpy(init).to(device).float()

    # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    #     start_time = time.time()
    #     with autocast():  # 开启混合精度计算
    #         ObjRecon = reConstruct(imstack, init, PSF)
    #     end_time = time.time()
    t1 = time.time()
    ObjRecon = reConstruct(imstack, init, PSF)
    t2 = time.time()
    print(t2-t1)

    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    tiff.imwrite('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/RLDoutput/output_00000004_RLD60.tif', ObjRecon)
