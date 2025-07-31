import os
import sys
import torch
from torchsummary import summary
from torchviz import make_dot
from datetime import datetime
import torchvision.transforms as transforms
import pytorch_lightning as pl
from dataset import CustomTiffDataset, compute_min_max, MinMaxNormalize
import torch.nn as nn

# 导入模型和损失函数
from utils.head import CombinedModel
from model.model import UNet_Transpose, UNet_Deeper
from loss import SSIMLoss, consistency_loss, consistency_loss_log

# 定义 PSNR 损失函数
class PSNRLoss(torch.nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return -psnr  # 返回负的 PSNR，因为我们要最小化它

# 定义 Lightning 模块
class UNetLightningModule(pl.LightningModule):
    def __init__(self, lf_extra, n_slices, output_size, learning_rate, input_min_val, input_max_val, gt_min_val, gt_max_val):
        super(UNetLightningModule, self).__init__()
        self.model = CombinedModel()  # 你的模型
        self.criterion = PSNRLoss()  # PSNR 损失
        self.SSIM_loss = SSIMLoss(size_average=True)
        self.consistency_loss_fn = consistency_loss
        self.mse_loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

# 加载模型
model_module = UNetLightningModule(lf_extra=27, n_slices=300, output_size=(600, 600), learning_rate=1e-4, input_min_val=0, input_max_val=1, gt_min_val=0, gt_max_val=1)

# 将模型移动到 GPU
model_module = model_module.cuda()

# 1. 使用 torchsummary 打印模型摘要
# print("Model Summary:")
# summary(model_module.model, input_size=(1, 28, 600, 600))  # 修改 input_size 为实际输入大小

# 2. 使用 torchviz 绘制模型计算图
# 输入一个示例数据
input_tensor = torch.rand(1, 28, 600, 600).cuda()  # 模拟输入，修改大小根据实际模型输入

# 前向传播，获得输出
output = model_module(input_tensor)

# 使用 torchviz 绘制模型的计算图
dot = make_dot(output, params=dict(model_module.model.named_parameters()))
dot.render("/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/network_arc/model_structure", format="png")  # 生成 model_structure.png 文件

