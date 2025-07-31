import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import CustomTiffDataset, compute_min_max, MinMaxNormalize
from model.model import UNet
from tqdm import tqdm
import tifffile as tiff
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
from loss import SSIMLoss  # 确保你的路径正确

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# 数据集路径
input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/input900'
gt_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt900'
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
label = str(current_time)
main_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/'
ckpt_dir = main_dir + 'ckpt/' + label
sample_dir = main_dir + 'sample/' + label
log_dir = main_dir + 'logs/' + label
os.makedirs(log_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

bz = 1
Ir = 1 * 1e-4
lf_extra = 27  # Number of input channels (example)
n_slices = 300  # Number of output slices
output_size = (600, 600)  # Output size

# 计算输入图像和标签的最小值和最大值
initial_transform = transforms.Compose([transforms.ToTensor()])
dataset = CustomTiffDataset(input_dir=input_dir, gt_dir=gt_dir, input_transform=initial_transform)
dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
input_min_val, input_max_val, gt_min_val, gt_max_val = compute_min_max(dataloader)

# 数据变换
input_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(input_min_val, input_max_val)
])
gt_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(gt_min_val, gt_max_val)
])

# 划分数据集为训练集和测试集
dataset = CustomTiffDataset(input_dir=input_dir, gt_dir=gt_dir, input_transform=input_transform,
                            gt_transform=gt_transform)
total_size = len(dataset)
test_size = int(0.1 * total_size)
train_size = total_size - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=False)

# 模型
model = UNet(lf_extra, n_slices, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
SSIM_loss = SSIMLoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=Ir)

# 初始化TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# 全局步数
global_step = 0
start_epoch = 0
best_loss = float('inf')

# 训练模型
num_epochs = 250
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_mse_loss = 0.0
    running_ssim_loss = 0.0
    running_total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for inputs, labels, _ in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # 零参数梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        mse_loss = criterion(outputs, labels)
        ssim_loss = SSIM_loss(outputs, labels)

        total_loss = mse_loss - torch.log((1 + ssim_loss) / 2)

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 统计损失
        running_mse_loss += mse_loss.item() * inputs.size(0)
        running_ssim_loss += ssim_loss.item() * inputs.size(0)
        running_total_loss += total_loss.item() * inputs.size(0)

        # 每50个global_step记录一次损失
        if global_step % 50 == 0:
            writer.add_scalar('Loss/train/total_loss_step', total_loss.item(), global_step)
            writer.add_scalar('Loss/train/ssim_loss_step', ssim_loss.item(), global_step)
            writer.add_scalar('Loss/train/mse_loss_step', mse_loss.item(), global_step)

        global_step += 1

    epoch_mse_loss = running_mse_loss / len(train_loader.dataset)
    epoch_ssim_loss = running_ssim_loss / len(train_loader.dataset)
    epoch_total_loss = running_total_loss / len(train_loader.dataset)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_total_loss:.8e}, MSE Loss: {epoch_mse_loss:.8e}, SSIM Loss: {epoch_ssim_loss:.8e}')

    # 记录每个epoch的训练损失
    writer.add_scalar('Loss/train/total_loss_epoch', epoch_total_loss, epoch)
    writer.add_scalar('Loss/train/ssim_loss_epoch', epoch_ssim_loss, epoch)
    writer.add_scalar('Loss/train/mse_loss_epoch', epoch_mse_loss, epoch)

    # 验证模型并保存推理结果
    model.eval()
    test_total_loss = 0.0
    test_ssim_loss = 0.0
    test_mse_loss = 0.0
    with torch.no_grad():
        for idx, (inputs, labels, _) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            mse_loss = criterion(outputs, labels)
            ssim_loss = SSIM_loss(outputs, labels)
            total_loss = mse_loss - torch.log((1 + ssim_loss) / 2)

            test_total_loss += total_loss.item() * inputs.size(0)
            test_ssim_loss += ssim_loss.item() * inputs.size(0)
            test_mse_loss += mse_loss.item() * inputs.size(0)

            # 每50个global_step记录一次验证损失
            if global_step % 50 == 0:
                writer.add_scalar('Loss/validation/total_loss_step', total_loss.item(), global_step)
                writer.add_scalar('Loss/validation/ssim_loss_step', ssim_loss.item(), global_step)
                writer.add_scalar('Loss/validation/mse_loss_step', mse_loss.item(), global_step)

            # 保存推理结果
            if (epoch + 1) % 10 == 1:
                original_idx = test_loader.dataset.indices[idx]
                input_filename = os.path.basename(dataset.image_files[original_idx])
                output_filename = f"{input_filename}_epoch{epoch + 1}.tif"
                output_path = os.path.join(sample_dir, output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image = outputs.cpu().numpy().squeeze()
                tiff.imwrite(output_path, image, compression="deflate")

            global_step += 1

    epoch_test_total_loss = test_total_loss / len(test_loader.dataset)
    epoch_test_ssim_loss = test_ssim_loss / len(test_loader.dataset)
    epoch_test_mse_loss = test_mse_loss / len(test_loader.dataset)

    print(
        f'Validation Total Loss: {epoch_test_total_loss:.8e}, MSE Loss: {epoch_test_mse_loss:.8e}, SSIM Loss: {epoch_test_ssim_loss:.8e}')
    if test_total_loss < best_loss:
        best_loss = test_total_loss
        best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved to {best_model_path} with loss {best_loss:.8e}")
    # 记录每个epoch的验证损失
    writer.add_scalar('Loss/validation/total_loss_epoch', epoch_test_total_loss, epoch)
    writer.add_scalar('Loss/validation/ssim_loss_epoch', epoch_test_ssim_loss, epoch)
    writer.add_scalar('Loss/validation/mse_loss_epoch', epoch_test_mse_loss, epoch)

# 保存模型
model_dir = os.path.join(log_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

print('训练完成，模型已保存。')
writer.close()
