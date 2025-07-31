import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import CustomTiffDataset, compute_min_max, MinMaxNormalize, CustomTiffDataset_double_gt, compute_min_max_double_gt
from model.model import UNet
from tqdm import tqdm
import tifffile as tiff
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from loss import sobel_edges, l2_loss, edges_loss, SSIMLoss, consistency_loss, consistency_loss_log
import h5py

# 设置设备为GPU 0和GPU 1
device_main = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_aux = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

# 数据集路径
input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/input900'
gt1_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt900'
gt2_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/input2_900'


# input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/debug_input'
# gt1_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/debug_gt'
# gt2_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/debug_input2'

label = str(current_time)
main_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/'
ckpt_dir = main_dir + 'ckpt/' + label
sample_dir = main_dir + 'sample/' + label
log_dir = main_dir + 'logs/' + label
#创建以上三个目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

data_label ='double_input900'

bz = 1
Ir = 1 * 1e-4

lf_extra = 27  # Number of input channels (example)
n_slices = 300  # Number of output slices
output_size = (600, 600)  # Output size

# 计算输入图像和标签的最小值和最大值
initial_transform = transforms.Compose([transforms.ToTensor()])
dataset = CustomTiffDataset_double_gt(input_dir=input_dir, gt_dir1=gt1_dir, gt_dir2=gt2_dir, input_transform=initial_transform)
dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
input_min_val, input_max_val, gt1_min_val, gt1_max_val, gt2_min_val, gt2_max_val = compute_min_max_double_gt(dataloader)

# 数据变换
input_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(input_min_val, input_max_val)
])
gt_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(gt1_min_val, gt1_max_val)
])

gt2_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(gt2_min_val, gt2_max_val)
])

# 划分数据集为训练集和测试集
dataset = CustomTiffDataset_double_gt(input_dir=input_dir, gt_dir1=gt1_dir, gt_dir2=gt2_dir, input_transform=input_transform,
                                      gt_transform=gt_transform,gt2_transform=gt2_transform)
total_size = len(dataset)
test_size = int(0.1 * total_size)
train_size = total_size - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=False)

# 模型
model = UNet(lf_extra, n_slices, output_size).to(device_main)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=Ir)

# 创建时间戳文件夹
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

os.makedirs(log_dir, exist_ok=True)

# 初始化TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# 全局步数
global_step = 0
start_epoch = 0
best_loss = float('inf')


def read_hdf5(h5_path, device):
    with h5py.File(h5_path, 'r') as f:
        matrix = f['PSF_1'][:]
    return torch.tensor(matrix, dtype=torch.float32, device=device)


PSF = read_hdf5('/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/RLD_test/PSF_G.mat', device_aux)

# 训练模型
num_epochs = 250
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_mse_loss = 0.0
    running_edges_loss = 0.0
    running_ssim_loss = 0.0
    running_consistency_loss = 0.0
    running_total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for inputs, labels, gt2_labels in progress_bar:
        inputs, labels, gt2_labels = inputs.to(device_main), labels.to(device_main), gt2_labels.to(device_aux)

        # 零参数梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        mse_loss = criterion(outputs, labels)
        edge_loss_value = edges_loss(outputs, labels)
        SSIM_loss = SSIMLoss(size_average=True)
        ssim_loss = SSIM_loss(outputs, labels)

        # consistency_loss 在 device_aux 上计算
        consistency_loss_value = consistency_loss(outputs.to(device_aux), gt2_labels.to(device_aux), PSF, device_aux)
        consistency_loss_value = consistency_loss_value.to(device_main)

        total_loss = mse_loss + 0.1 * edge_loss_value - torch.log((1 + ssim_loss) / 2) + (1*1e-12)*consistency_loss_value

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 统计损失
        running_mse_loss += mse_loss.item() * inputs.size(0)
        running_edges_loss += edge_loss_value.item() * inputs.size(0)
        running_ssim_loss += ssim_loss.item() * inputs.size(0)
        running_consistency_loss += consistency_loss_value.item() * inputs.size(0)
        running_total_loss += total_loss.item() * inputs.size(0)

        global_step += 1

    epoch_mse_loss = running_mse_loss / len(train_loader.dataset)
    epoch_edges_loss = running_edges_loss / len(train_loader.dataset)
    epoch_ssim_loss = running_ssim_loss / len(train_loader.dataset)
    epoch_consistency_loss = running_consistency_loss / len(train_loader.dataset)
    epoch_total_loss = running_total_loss / len(train_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_total_loss:.8e}, Edges Loss: {epoch_edges_loss:.8e}, Mse Loss: {epoch_mse_loss:.8e}, SSIM Loss: {epoch_ssim_loss:.8e}, Consistency Loss: {epoch_consistency_loss:.8e}')

    writer.add_scalar('Loss/train/total_loss_epoch', epoch_total_loss, epoch)
    writer.add_scalar('Loss/train/edge_loss_epoch', epoch_edges_loss, epoch)
    writer.add_scalar('Loss/train/ssim_loss_epoch', epoch_ssim_loss, epoch)
    writer.add_scalar('Loss/train/mse_loss_epoch', epoch_mse_loss, epoch)
    writer.add_scalar('Loss/train/consistency_loss_epoch', epoch_consistency_loss, epoch)

    model.eval()
    test_total_loss = 0.0
    test_edges_loss = 0.0
    test_mse_loss = 0.0
    test_ssim_loss = 0.0
    test_consistency_loss = 0.0
    with torch.no_grad():
        for idx, (inputs, labels, gt2_labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, labels, gt2_labels = inputs.to(device_main), labels.to(device_main), gt2_labels.to(device_aux)
            outputs = model(inputs)
            mse_loss = criterion(outputs, labels)
            edge_loss_value = edges_loss(outputs, labels)
            ssim_loss = SSIM_loss(outputs, labels)
            consistency_loss_value = consistency_loss_log(outputs.to(device_aux), gt2_labels, PSF, device_aux, global_step, writer)
            consistency_loss_value = consistency_loss_value.to(device_main)

            total_loss = mse_loss + 0.1 * edge_loss_value - torch.log((1 + ssim_loss) / 2) + (1*1e-12)*consistency_loss_value

            test_total_loss += total_loss.item() * inputs.size(0)
            test_edges_loss += edge_loss_value.item() * inputs.size(0)
            test_mse_loss += mse_loss.item() * inputs.size(0)
            test_ssim_loss += ssim_loss.item() * inputs.size(0)
            test_consistency_loss += consistency_loss_value.item() * inputs.size(0)

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
    epoch_test_edges_loss = test_edges_loss / len(test_loader.dataset)
    epoch_test_mse_loss = test_mse_loss / len(test_loader.dataset)
    epoch_test_ssim_loss = test_ssim_loss / len(test_loader.dataset)
    epoch_test_consistency_loss = test_consistency_loss / len(test_loader.dataset)

    print(f'Validation Total Loss: {epoch_test_total_loss:.8e}, Edges Loss: {epoch_test_edges_loss:.8e}, Mse Loss: {epoch_test_mse_loss:.8e}, SSIM Loss: {epoch_test_ssim_loss:.8e}, Consistency Loss: {epoch_test_consistency_loss:.8e}')
    if test_total_loss < best_loss:
        best_loss = test_total_loss
        best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved to {best_model_path} with loss {best_loss:.8e}")

    writer.add_scalar('Loss/validation/total_loss_epoch', epoch_test_total_loss, epoch)
    writer.add_scalar('Loss/validation/edges_loss_epoch', epoch_test_edges_loss, epoch)
    writer.add_scalar('Loss/validation/mse_loss_epoch', epoch_test_mse_loss, epoch)
    writer.add_scalar('Loss/validation/ssim_loss_epoch', epoch_test_ssim_loss, epoch)
    writer.add_scalar('Loss/validation/consistency_loss_epoch', epoch_test_consistency_loss, epoch)

model_dir = os.path.join(log_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

print('训练完成，模型已保存。')
writer.close()
