
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from datetime import datetime


import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
from dataset import CustomTiffDataset, compute_min_max, MinMaxNormalize,CustomTiffDataset_FP, SynchronizedTransform, ValTransform
current_dir = os.path.dirname(__file__)
# 获取src目录的路径
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
import h5py
# 将src目录添加到sys.path
sys.path.append(src_dir)
from utils.head import CombinedModel_forward
# 导入模型
from model.model import UNet_Transpose, UNet_Deeper
from loss import SSIMLoss, consistency_loss, consistency_loss_log
import tifffile as tiff
import time
import torch.autograd.profiler as profiler
# PSNR Loss function
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Subset


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
        self.model = CombinedModel_forward()
        self.criterion = PSNRLoss()
        self.SSIM_loss = SSIMLoss(size_average=True)
        self.consistency_loss_fn = consistency_loss
        self.mse_loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val
        #h5_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/RLD_test/PSF_G.mat'

        #self.PSF = self.read_hdf5(h5_path)
        self.relu = nn.ReLU()



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




        outputs = self(inputs)
        #记录图像
        # if self.global_step % 100 == 0:
        #     self.logger.experiment.add_image('Train/Output', outputs[0], self.global_step)
        #     self.logger.experiment.add_image('Train/Label', labels[0], self.global_step)

        if self.global_step % 1 == 0:
            self.logger.experiment.add_image('Train/Input', inputs[:,150,:,:]*self.input_max_val, self.global_step)
            self.logger.experiment.add_image('Train/Output', outputs[:,13,:,:]*self.gt_max_val, self.global_step)
            self.logger.experiment.add_image('Train/Label', labels[:,13,:,:]*self.gt_max_val, self.global_step)

        # 计算 PSNR 损失
        psnr_loss = self.criterion(outputs, labels)

        # 调用 consistency_loss 函数，并传递 writer 和 global_step
        #consistency_loss_value = consistency_loss_log(outputs, labels, self.PSF_fft, self.logger, self.global_step)

        # 计算总损失
        total_loss = (psnr_loss / 60)

        # 记录损失到 TensorBoard
        self.log('train_loss', total_loss,sync_dist=True)
        self.log('train_psnr_loss', psnr_loss,sync_dist=True)
        #self.log('train_consistency_loss', consistency_loss_value,sync_dist=True)

        # 每 100 个 batch 记录一次输出和标签图像到 TensorBoard
        # if self.global_step % 100 == 0:
        #     self.logger.experiment.add_image('Train/Output', outputs[0], self.global_step)
        #     self.logger.experiment.add_image('Train/Label', labels[0], self.global_step)

        # 更新 global_step


        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch

        outputs = self(inputs)
        if self.global_step % 1 == 0:
            self.logger.experiment.add_image('Validation/Input', inputs[:, 150, :, :] * self.input_max_val, self.global_step)
            self.logger.experiment.add_image('Validation/Output', outputs[:, 13, :, :] * self.gt_max_val, self.global_step)
            self.logger.experiment.add_image('Validation/Label', labels[:, 13, :, :] * self.gt_max_val, self.global_step)



        mse_loss = self.mse_loss_fn(outputs, labels)
        ssim_loss = self.SSIM_loss(outputs, labels)
        #consistency_loss_value = self.consistency_loss_fn(outputs, labels, self.PSF_fft)
        total_loss = mse_loss
        psnr_loss = self.criterion(outputs, labels)
        # Modify this if you need to combine with SSIM or other losses
        self.log('val_loss', total_loss)
        self.log('val_mse_loss', mse_loss)
        self.log('val_ssim_loss', ssim_loss)
        #self.log('val_consistency_loss', consistency_loss_value)
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
                    tiff.imwrite(output_path, image*self.gt_max)
            pl_module.model.train()

# Data Module
def read_file_list(file_path):
    """Reads the list of filenames from a text file."""
    with open(file_path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


# Data Module
class TiffDataModule(pl.LightningDataModule):
    def __init__(self, input_dir, gt_dir, batch_size, synchronized_transform, val_transform,
                 train_file_list_path, val_file_list_path):
        super().__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.synchronized_transform = synchronized_transform
        self.val_transform = val_transform
        self.train_file_list_path = train_file_list_path
        self.val_file_list_path = val_file_list_path

    def setup(self, stage=None):
        # Read the file lists from txt files
        train_file_list = read_file_list(self.train_file_list_path)
        val_file_list = read_file_list(self.val_file_list_path)

        # Create datasets with file lists
        self.train_dataset = CustomTiffDataset_FP(input_dir=self.input_dir, gt_dir=self.gt_dir,
                                                  file_list=train_file_list, transform=self.synchronized_transform)

        self.val_dataset = CustomTiffDataset_FP(input_dir=self.input_dir, gt_dir=self.gt_dir,
                                                file_list=val_file_list, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, pin_memory=True)


# Main script
if __name__ == "__main__":
    # Parameters and paths
    input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD60_1500'
    gt_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD60_1500_conv_crop'

    # File list paths
    train_file_list_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/data_div/RLD60_1500/train.txt'
    val_file_list_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/data_div/RLD60_1500/val.txt'

    # Other necessary parameters
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tag = 'train_FPmodel_low_pure_PSNR_conv_RLD60_fish1_1500_0915'
    label = tag + str(current_time)
    checkpoint_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt/NAFNet_deeper_relu_PSNR_conv_RLD60_fish1_total_0910_with_low32_pretrained_aug20240910-005034/epoch=149-val_loss=0.0000000102.ckpt'
    main_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/'
    ckpt_dir = os.path.join(main_dir, 'ckpt', label)
    sample_dir = os.path.join(main_dir, 'sample', label)
    log_dir = os.path.join(main_dir, 'logs', label)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    bz = 1
    Ir = 1 * 1e-4
    lf_extra = 27  # Number of input channels
    n_slices = 300  # Number of output slices
    output_size = (600, 600)  # Output size

    # Load normalization values from pre-saved file
    input_min_val, input_max_val, gt_min_val, gt_max_val = np.load(
        '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/FP_fish1_1500.npy')

    # Create synchronized transform for training
    synchronized_transform = SynchronizedTransform(input_min_val, input_max_val, gt_min_val, gt_max_val)

    # Create val transform (no augmentation, just normalization)
    val_transform = ValTransform(input_min_val, input_max_val, gt_min_val, gt_max_val)

    # Data module with file lists
    data_module = TiffDataModule(
        input_dir=input_dir,
        gt_dir=gt_dir,
        batch_size=bz,
        synchronized_transform=synchronized_transform,
        val_transform=val_transform,
        train_file_list_path=train_file_list_path,
        val_file_list_path=val_file_list_path
    )
    # Model module
    checkpoint = torch.load(checkpoint_path)
    model_module = UNetLightningModule(lf_extra=lf_extra, n_slices=n_slices, output_size=output_size,
                                       learning_rate=Ir, input_min_val=input_min_val, input_max_val=input_max_val,
                                       gt_min_val=gt_min_val, gt_max_val=gt_max_val)

    # 获取模型的 state_dict
    model_state_dict = model_module.state_dict()

    # 对比检查点的 state_dict 和当前模型的 state_dict，只加载匹配的部分
    checkpoint_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                             k in model_state_dict and model_state_dict[k].shape == v.shape}
    #
    # # 更新模型参数
    model_state_dict.update(checkpoint_state_dict)
    model_module.load_state_dict(model_state_dict)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename='{epoch}-{val_loss:.10f}'
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    save_inference_callback = SaveInferenceCallback(sample_dir, epoch_interval=10, gt_max=gt_max_val)

    # Logger
    logger = pl.loggers.TensorBoardLogger(log_dir)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=250,
        devices=1,
        accelerator='gpu',
        strategy='ddp',
        callbacks=[checkpoint_callback, lr_monitor, save_inference_callback],
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        precision=16,
    )

    # Train the model
    trainer.fit(model_module, datamodule=data_module)