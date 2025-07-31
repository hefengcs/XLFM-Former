import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import tifffile as tiff
from tqdm import tqdm
import numpy as np
current_dir = os.path.dirname(__file__)
# 获取src目录的路径
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
import sys
sys.path.append(src_dir)
from dataset import MinMaxNormalize
from utils.head import CombinedModel_deeper
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义推理数据集
class InferenceTiffDataset(Dataset):
    def __init__(self, input_dir, input_transform=None):
        self.input_dir = input_dir
        self.input_transform = input_transform
        self.image_files = os.listdir(input_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image = np.transpose(input_image, (1, 2, 0))

        if self.input_transform:
            input_image = self.input_transform(input_image)

        return input_image, img_name  # 返回文件名


# 加载模型
lf_extra = 27  # Number of input channels (example)
n_slices = 300  # Number of output slices
output_size = (600, 600)  # Output size

model = CombinedModel_deeper().to(device)
model_ckpt = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt/NAFNet_deeper_relu_PSNR_conv_RLD60_fish1_1500_0907_with_low_pretrained_aug20240907-172242/epoch=234-val_loss=0.0000005459.ckpt'  # 指定模型权重文件路径

# 加载 PyTorch Lightning 检查点并去除前缀
checkpoint = torch.load(model_ckpt, map_location=device)
state_dict = checkpoint['state_dict']
new_state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# 如果有多张 GPU，则使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for inference")
    model = torch.nn.DataParallel(model)

input_min_val, input_max_val, gt_min_val, gt_max_val = np.load(
    '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/fixed_fish3.npy')

# 数据变换
input_transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize(input_min_val, input_max_val)
])

# 推理数据集路径
input_dir = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/240725-03/input_location'
output_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/inference/fixed_fish3'

# 加载推理数据集
dataset = InferenceTiffDataset(input_dir=input_dir, input_transform=input_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# Denormalize 函数



# 推理
model.eval()
with torch.no_grad():
    for input_image, file_name in tqdm(dataloader, desc="Inference"):
        input_image = input_image.to(device)

        # 在多GPU环境下，DataParallel 会将数据自动分配到多个GPU上
        outputs = model(input_image)

        # 将输出移回 CPU 并进行后处理
        outputs = outputs.squeeze(0)
        output_path = os.path.join(output_dir, f'infer{file_name[0]}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np_output = outputs.cpu().numpy().squeeze()
        np_output[np_output < 0] = 0

        # 反归一化
        np_output =np_output*gt_max_val

        # 保存为 TIFF 文件
        tiff.imwrite(output_path, np_output)

print('推理完成，结果已保存。')
