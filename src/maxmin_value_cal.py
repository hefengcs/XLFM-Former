import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
import sys
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
from dataset import CustomTiffDataset, compute_min_max, MinMaxNormalize
current_dir = os.path.dirname(__file__)
# 获取src目录的路径
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
import h5py
# 将src目录添加到sys.path
sys.path.append(src_dir)
from utils.head import CombinedModel_deeper_w32
# 导入模型
from model.model import UNet_Transpose, UNet_Deeper
from loss import SSIMLoss, consistency_loss, consistency_loss_log
import tifffile as tiff
import time
import torch.autograd.profiler as profiler
# PSNR Loss function
import torch.nn as nn
import torchvision.transforms.functional as F



class PSNRLoss(torch.nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return -psnr  # Note: We return negative PSNR for minimization

# Lightning Module
class UNetLightningModule(pl.LightningModule):
    def __init__(self, lf_extra, n_slices, output_size, learning_rate, input_min_val, input_max_val, gt_min_val,
                 gt_max_val):
        super(UNetLightningModule, self).__init__()
        self.model = CombinedModel_deeper_w32()
        self.criterion = PSNRLoss()
        self.SSIM_loss = SSIMLoss(size_average=True)
        self.consistency_loss_fn = consistency_loss
        self.mse_loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val
        h5_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/RLD_test/PSF_G.mat'

        self.PSF = self.read_hdf5(h5_path)
        self.relu = nn.ReLU()

    def read_hdf5(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            matrix = f['PSF_1'][:]
            matrix = torch.tensor(matrix, dtype=torch.float32)
            matrix = matrix.permute(0, 2, 1)
        return matrix

    def pad_to_target(self, matrix, target_size):
        matrix = matrix.squeeze(0)
        padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=matrix.device)
        z_start = (target_size[0] - matrix.size(0)) // 2
        y_start = (target_size[1] - matrix.size(1)) // 2
        x_start = (target_size[2] - matrix.size(2)) // 2

        padded_matrix[z_start:z_start + matrix.size(0),
        y_start:y_start + matrix.size(1),
        x_start:x_start + matrix.size(2)] = matrix
        return padded_matrix

    def setup(self, stage=None):
        # 在这里确保 PSF 的设备与模型一致
        # def ifftshift(x):
        #     shift = [-(dim // 2) for dim in x.shape]
        #     return torch.roll(x, shift, dims=(-2, -1))
        def ifftshift(x):
            shift = [-(dim // 2) for dim in x.shape[-2:]]  # 只计算最后两个维度的shift
            return torch.roll(x, shift, dims=(-2, -1))  # 在最后两个维度上进行roll操作


        def fftshift(x):
            shift = [dim // 2 for dim in x.shape]
            return torch.roll(x, shift, dims=(-2, -1))
        self.PSF = self.PSF.to(self.device)
        self.PSF = self.pad_to_target(self.PSF, (300, 2304, 2304))
        #先进行中心化
        self.PSF = ifftshift(self.PSF)
        self.PSF_fft = torch.fft.fft2(self.PSF)
        self.PSF_fft = self.PSF_fft.to(self.device)
        #清除self.PSF的显存
        self.PSF = None

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch

        # 模型前向传播
        outputs = self(inputs)

        # 计算 PSNR 损失
        psnr_loss = self.criterion(outputs, labels)

        # 调用 consistency_loss 函数，并传递 writer 和 global_step
        consistency_loss_value = consistency_loss_log(outputs, labels, self.PSF_fft, self.logger, self.global_step)

        # 计算总损失
        total_loss = (psnr_loss / 60) + (consistency_loss_value / 1e-7)

        # 记录损失到 TensorBoard
        self.log('train_loss', total_loss,sync_dist=True)
        self.log('train_psnr_loss', psnr_loss,sync_dist=True)
        self.log('train_consistency_loss', consistency_loss_value,sync_dist=True)

        # 每 100 个 batch 记录一次输出和标签图像到 TensorBoard
        # if self.global_step % 100 == 0:
        #     self.logger.experiment.add_image('Train/Output', outputs[0], self.global_step)
        #     self.logger.experiment.add_image('Train/Label', labels[0], self.global_step)

        # 更新 global_step


        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        mse_loss = self.mse_loss_fn(outputs, labels)
        ssim_loss = self.SSIM_loss(outputs, labels)
        consistency_loss_value = self.consistency_loss_fn(outputs, labels, self.PSF_fft)
        total_loss = mse_loss
        psnr_loss = self.criterion(outputs, labels)
        # Modify this if you need to combine with SSIM or other losses
        self.log('val_loss', total_loss)
        self.log('val_mse_loss', mse_loss)
        self.log('val_ssim_loss', ssim_loss)
        self.log('val_consistency_loss', consistency_loss_value)
        self.log('val_psnr_loss', psnr_loss)
        return mse_loss

# Custom callback to save inference results every 10 epochs
class SaveInferenceCallback(pl.Callback):
    def __init__(self, sample_dir, gt_max,epoch_interval=20,):
        self.sample_dir = sample_dir
        self.epoch_interval = epoch_interval
        self.gt_max = gt_max
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.epoch_interval == 0:
            val_loader = trainer.datamodule.val_dataloader()
            pl_module.model.eval()
            with torch.no_grad():
                for idx, (inputs, labels, input_filename) in enumerate(val_loader):
                    inputs = inputs.to(pl_module.device)
                    outputs = pl_module(inputs)
                    #input_filename = f"input_{idx}.tif"
                    output_filename = f"{input_filename}_epoch{epoch + 1}.tif"
                    output_path = os.path.join(self.sample_dir, output_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    image = outputs.cpu().numpy().squeeze()
                    tiff.imwrite(output_path, image*self.gt_max, compression="deflate")
            pl_module.model.train()

# Data Module
class TiffDataModule(pl.LightningDataModule):
    def __init__(self, input_dir, gt_dir, batch_size, train_input_transform, train_gt_transform, val_input_transform, val_gt_transform):
        super().__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.train_input_transform = train_input_transform
        self.train_gt_transform = train_gt_transform
        self.val_input_transform = val_input_transform
        self.val_gt_transform = val_gt_transform

    def setup(self, stage=None):
        dataset = CustomTiffDataset(input_dir=self.input_dir, gt_dir=self.gt_dir, input_transform=self.train_input_transform, gt_transform=self.train_gt_transform)
        total_size = len(dataset)
        test_size = int(0.1 * total_size)
        train_size = total_size - test_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, test_size])

        # Validation dataset should use non-augmented transformations
        self.val_dataset.dataset.input_transform = self.val_input_transform
        self.val_dataset.dataset.gt_transform = self.val_gt_transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

# Main script
if __name__ == "__main__":
    # Parameters and paths
    input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_learning/input1500'
    gt_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_learning/gt1500'
    save_path = '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/psf_learning_beads_1500.npy'

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tag = 'NAFNet_deeper_relu_PSNR_conv_RLD60_fish1_1500_0907_with_low_pretrained_aug'
    label = tag + str(current_time)
    checkpoint_path = '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt/NAFNet_relu_PSNR_conv_RLD60_fish1_5000_0903_with_pretrained20240903-184716/epoch=201-val_loss=0.0000008170.ckpt'
    main_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/'
    ckpt_dir = os.path.join(main_dir, 'ckpt', label)
    sample_dir = os.path.join(main_dir, 'sample', label)
    log_dir = os.path.join(main_dir, 'logs', label)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    bz = 1
    #Ir = 1 * 1e-4
    Ir = 1 * 1e-4
    lf_extra = 27  # Number of input channels
    n_slices = 300  # Number of output slices
    output_size = (600, 600)  # Output size

    # Initial transform for computing min and max values
    initial_transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomTiffDataset(input_dir=input_dir, gt_dir=gt_dir, input_transform=initial_transform)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True,num_workers=64 )
    input_min_val, input_max_val, gt_min_val, gt_max_val = compute_min_max(dataloader,save_path=save_path)