import os

import argparse
import torch
import numpy as np
import tifffile as tiff
import h5py
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from tqdm import tqdm


class ReconstructionDataset(Dataset):
    def __init__(self, input_dir, start_file=None):
        self.files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.endswith('.tif')]
        if start_file:
            start_file_path = os.path.join(input_dir, start_file)
            if start_file_path in self.files:
                start_index = self.files.index(start_file_path)
                self.files = self.files[start_index:]  # Start processing from specified file
            else:
                print(f"Start file {start_file} not found in the directory. Processing all files.")
        else:
            print("No start file specified. Processing all files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imstack = tiff.imread(self.files[idx]).astype(np.float32)
        return torch.from_numpy(imstack), self.files[idx]


import torch
import numpy as np


class RichardsonLucySubModule(torch.nn.Module):
    def __init__(self, init, PSF, device_id):
        super().__init__()
        self.init = init.to(device_id)
        self.PSF = PSF.to(device_id)
        self.device_id = device_id

        # Precompute the FFT of the PSF and its conjugate
        NxyExt = 128
        Nxy = 2304
        Nz = 300

        # Pad the PSF
        self.PSF_padded = torch.nn.functional.pad(self.PSF, (0, 0, NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

        # Precompute FFT of the PSF slices and their conjugates
        self.fft_PSF = []
        self.fft_PSF_conj = []
        for jj in range(Nz):
            PSF_jj = self.PSF_padded[:, :, jj]
            fft_PSF_jj = torch.fft.fft2(torch.fft.ifftshift(PSF_jj))
            self.fft_PSF.append(fft_PSF_jj)
            self.fft_PSF_conj.append(torch.conj(fft_PSF_jj))

    def forward(self, imstack):
        ItN = 60  # 迭代次数
        BkgMean = 110  # 背景噪声平均水平
        ROISize = 300  # 感兴趣区域大小
        SNR = 200  # 信噪比
        NxyExt = 128  # 图像扩展大小
        Nxy = 2304  # 扩展后图像大小
        Nz = 300  # Z轴高度

        gpuObjReconTmp = torch.zeros((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        ImgEst = torch.zeros((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        Ratio = torch.ones((Nxy, Nxy), device=self.device_id, dtype=torch.float32)
        Img = imstack.to(self.device_id)
        ImgMultiView = Img - BkgMean
        ImgMultiView = torch.clamp(ImgMultiView, min=0)
        ImgExp = torch.nn.functional.pad(ImgMultiView, (NxyExt, NxyExt, NxyExt, NxyExt), 'constant', 0)

        gpuObjRecon = self.init.clone()

        for ii in range(ItN):
            ImgEst.fill_(0)
            for jj in range(Nz):
                gpuObjReconTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                Nxy // 2 - ROISize:Nxy // 2 + ROISize] = gpuObjRecon[jj, :, :]

                ImgEst_slice_update = torch.real(
                    torch.fft.ifft2(torch.fft.fft2(gpuObjReconTmp) * self.fft_PSF[jj])) / torch.sum(
                    self.PSF_padded[:, :, jj])
                ImgEst += torch.max(ImgEst_slice_update, torch.tensor(0.0, device=ImgEst_slice_update.device))

            ImgExpEst = ImgExp

            Tmp = torch.median(ImgEst)

            Ratio.fill_(1)
            Ratio[NxyExt:-NxyExt, NxyExt:-NxyExt] = ImgExpEst[NxyExt:-NxyExt, NxyExt:-NxyExt] / (
                    ImgEst[NxyExt:-NxyExt, NxyExt:-NxyExt] + Tmp / SNR)

            for jj in range(Nz):
                fft_Ratio = torch.fft.fft2(Ratio)

                ifft_result = torch.fft.ifft2(fft_Ratio * self.fft_PSF_conj[jj])

                gpuTmp = torch.real(ifft_result) / torch.sum(self.PSF_padded[:, :, jj])

                gpuTmp = torch.maximum(gpuTmp, torch.tensor(0.0, device=self.device_id))

                gpuObjRecon[jj, :, :] *= gpuTmp[Nxy // 2 - ROISize:Nxy // 2 + ROISize,
                                         Nxy // 2 - ROISize:Nxy // 2 + ROISize]

        ObjRecon = gpuObjRecon.cpu().numpy().astype(np.float32)
        return ObjRecon


def worker(device_id, data_queue, result_queue, init, PSF):
    torch.cuda.set_device(device_id)
    model = RichardsonLucySubModule(init, PSF, device_id)
    with torch.no_grad():
        while True:
            item = data_queue.get()
            if item is None:  # None indicates no more data
                break
            data, file_path = item
            # 移除第一个维度
            data = data.squeeze(0)

            output = model(data.cuda(device_id))
            result_queue.put((output, file_path))  # 确保 file_path 是字符串


def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--start_file', type=str, help='The file to start processing from', default=None)
    parser.add_argument('--cuda_devices', type=str, help='CUDA visible devices', default='0,5')
    args = parser.parse_args()

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    init = np.ones((300, 600, 600), dtype=np.float32)
    with h5py.File('/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_G.mat', 'r') as f:
        PSF = f['PSF_1'][:]
    PSF = PSF.astype(np.float32)
    PSF = np.transpose(PSF, (2, 1, 0))

    input_dir = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/Recon/2023-11-11_17-57-58_g8s_lssm_6dpf/g'
    output_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/fish1_gt_RLD60'
    start_file = args.start_file

    dataset = ReconstructionDataset(input_dir, start_file)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    num_gpus = 8

    # 创建进程间通信队列
    data_queue = mp.Queue()
    result_queue = mp.Queue()

    # 启动子进程
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(i, data_queue, result_queue, torch.from_numpy(init), torch.from_numpy(PSF)))
        p.start()
        processes.append(p)

    # 向队列中加入数据
    for data, file_path in data_loader:
        data_queue.put((data, file_path))  # 确保 file_path 是一个字符串

    # 向子进程发送终止信号
    for _ in range(num_gpus):
        data_queue.put(None)

    # 处理结果并保存
    progress_bar = tqdm(total=len(dataset), desc="Overall Progress", position=0)
    while progress_bar.n < len(dataset):
        output, file_path = result_queue.get()

        output_path = os.path.join(output_dir, os.path.basename(file_path[0]))  # 解包元组，获取正确的 file_path
        tiff.imwrite(output_path, output)
        progress_bar.update(1)

    for p in processes:
        p.join()

    progress_bar.close()


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Needed for CUDA operations
    main()
