import os
import tifffile as tiff
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import random
from torchvision.transforms import functional as TF
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torchvision.transforms import functional as TF
import os
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random
torch.multiprocessing.set_sharing_strategy('file_system')
# 自定义的随机翻转和旋转类
class RandomFlipRotate:
    def __call__(self, img, gt):
        # 随机选择翻转类型和旋转角度
        flip_type = random.choice(['horizontal', 'vertical', 'none'])
        rotate_degree = random.choice([0, 90, 180, 270])

        # 应用翻转
        if flip_type == 'horizontal':
            img = TF.hflip(img)
            gt = TF.hflip(gt)
        elif flip_type == 'vertical':
            img = TF.vflip(img)
            gt = TF.vflip(gt)

        # 应用旋转
        img = TF.rotate(img, rotate_degree)
        gt = TF.rotate(gt, rotate_degree)

        return img, gt
from pathlib import Path
# 自定义数据集类，支持数据增强
class CustomTiffDatasetWithAugmentation(Dataset):
    def __init__(self, input_dir, gt_dir, input_transform=None, gt_transform=None, augmentation=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.augmentation = augmentation  # Augmentation operation
        self.image_files = os.listdir(input_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image = np.transpose(input_image, (1, 2, 0))

        # 加载对应的标签图像
        gt_path = os.path.join(self.gt_dir, img_name)
        gt_image = tiff.imread(gt_path).astype(np.float32)
        gt_image = np.transpose(gt_image, (1, 2, 0))

        # 如果定义了数据增强，应用到输入图像和GT图像 (仅在训练时)
        if self.augmentation:
            input_image, gt_image = self.augmentation(input_image, gt_image)

        # 应用其他输入和GT变换
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.gt_transform:
            gt_image = self.gt_transform(gt_image)

        return input_image, gt_image, img_name
class CustomTiffDataset(Dataset):
    def __init__(self, input_dir, gt_dir, input_transform=None, gt_transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.image_files = os.listdir(input_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image =np.transpose(input_image,(1,2,0))
        # 加载对应的标签
        gt_path = os.path.join(self.gt_dir, img_name)
        gt_image = tiff.imread(gt_path).astype(np.float32)
        gt_image = np.transpose(gt_image,(1,2,0))
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.gt_transform:
            gt_image = self.gt_transform(gt_image)

        return input_image, gt_image,img_name


def normalize_percentile(im, low=0.5, high=99.5):
    p_low = np.percentile(im, low)
    p_high = np.percentile(im, high)
    eps = 1e-3  # 防止除以0
    normalized_im = (im - p_low) / (p_high - p_low + eps)
    normalized_im = np.clip(normalized_im, 0, 1)  # 裁剪到 [0, 1] 范围
    return normalized_im, p_low, p_high
class View_mask_dataset(Dataset):
    def __init__(self, txt_file, train_transform=None, val_transform=None, raw=False,ratio=0.75,ratio_pattern="View"):
        """
        初始化数据集
        :param txt_file: 包含文件路径的txt文件
        :param train_transform: 训练集数据增强
        :param val_transform: 验证集数据增强
        :param raw: 是否对输入进行特殊归一化处理
        """
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.raw = raw
        self.ratio =ratio
        self.ratio_pattern = ratio_pattern
        # 从 txt 文件中读取要加载的文件名
        with open(txt_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        input_path = self.image_files[idx]
        _, img_name = os.path.split(input_path)

        # 读取 tiff 图像
        input_image = tiff.imread(input_path).astype(np.float32)

        # 百分位归一化的范围
        p_low = 0
        p_high = 2000

        # Step 1: 归一化输入图像
        input_image_norm = (input_image - p_low) / (p_high - p_low)
        input_image_norm = np.clip(input_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围

        if self.raw:
            input_image_norm = input_image_norm[np.newaxis, :, :]  # shape (1, H, W)
            input_image_norm = np.repeat(input_image_norm, 3, axis=0)  # shape (3, H, W)

        # Step 2: 调整输入图像的形状到 (28, 600, 600)
        input_image_norm = input_image_norm[np.newaxis, :, :]  # shape (1, H, W)
        #input_image_norm = np.repeat(input_image_norm, 28, axis=0)  # shape (28, H, W)

        # Step 3: 随机 mask 掉 30% 的视角
        #
        if self.ratio_pattern == "view":
            input_image_masked = self.randomly_zero_out_layers(input_image_norm, drop_rate=self.ratio)



        #使用完全随机的掩码策略
        if self.ratio_pattern == "random":
            input_image_masked = self.randomly_zero_out_pixels(input_image_norm, drop_rate=self.ratio)

        # Step 4: 应用自定义的训练或验证变换
        if self.train_transform is not None:
            input_image_masked = self.train_transform(input_image_masked)

        if self.val_transform is not None:
            input_image_masked = self.val_transform(input_image_masked)

        # 将 masked 的输入作为 `gt`
        gt_image_norm = input_image_norm

        #去掉0维度
        input_image_masked = input_image_masked.squeeze(0)
        gt_image_norm = gt_image_norm.squeeze(0)

        return input_image_masked, gt_image_norm, img_name, p_low, p_high
    @staticmethod
    def randomly_zero_out_pixels(input_image, drop_rate=0.3):
        """
        随机丢弃（置0）输入数组的像素
        :param input_image: 输入的numpy数组，维度为 (C, H, W)
        :param drop_rate: 每个像素被丢弃（置0）的概率，默认为30%
        :return: 处理后的数组
        """
        mask = np.random.rand(*input_image.shape) >= drop_rate  # 生成与输入形状相同的随机掩码
        output_image = input_image * mask  # 应用掩码
        return output_image


    @staticmethod
    def randomly_zero_out_layers(input_image, drop_rate=0.3):
        """
        随机丢弃（置0）输入数组的层
        :param input_image: 输入的numpy数组，维度为 (28, H, W)
        :param drop_rate: 每一层被丢弃（置0）的概率，默认为30%
        :return: 丢弃后的数组
        """
        output_image = input_image.copy()  # 创建副本，避免修改原始数据
        num_layers = output_image.shape[1]  # 获取层数

        # 遍历每一层
        for layer in range(num_layers):
            if random.random() < drop_rate:
                output_image[:,layer, :, :] = 0  # 将该层置为0

        return output_image

# 自定义数据集类
class CustomTiffDatasetSinglePercentile(Dataset):
    #def __init__(self, input_dir, gt_dir, txt_file, input_transform=None, gt_transform=None):
    def __init__(self,  txt_file, train_transform=None,val_transform=None,raw=False,view_ratio = 0):
        # self.input_dir = input_dir
        # self.gt_dir = gt_dir
        # self.input_transform = input_transform
        # self.gt_transform = gt_transform
        self.train_transform = train_transform
        self.val_transform =val_transform
        self.raw = raw
        self.view_ratio = view_ratio
        # 从 txt 文件中读取要加载的文件名
        with open(txt_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        input_path = self.image_files[idx]

        _, img_name=os.path.split(input_path)

        input_image = tiff.imread(input_path).astype(np.float32)
        #input_image = np.transpose(input_image, (1, 2, 0))  # 转置通道顺序 (H, W, C)

        input_path =Path(input_path)
        input_location = input_path.parents[0].name
        # 加载对应的标签图像
        gt_path =str(input_path).replace(str(input_location), "RLD60")
        gt_image = tiff.imread(gt_path).astype(np.float32)
        #gt_image = np.transpose(gt_image, (1, 2, 0))  # 转置通道顺序 (H, W, C)

        # Step 1: 对输入图像计算百分位并归一化
        if self.raw == True:
            input_image = input_image-160
            #截断一下
            input_image[input_image < 0] = 0
        p_low = 0
        p_high = 2000
        input_image_norm =(input_image - p_low) / (p_high - p_low )
        input_image_norm = np.clip(input_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围

        #用于Nerf:总共有28层，将7，11，12，28设置为全0，其他设置为正常的
        # channels_to_zero = [6, 10, 11, 27]  # 对应于第7,11,12,28通道（从0开始）
        # input_image_norm[channels_to_zero] = 0

        if self.raw == True:
            #增加一个0维度
            input_image_norm = input_image_norm[np.newaxis, :, :]  # shape (1, 600, 600)
            #0通道重复三次，维度变成3，2048，2048
            input_image_norm = np.repeat(input_image_norm, 3, axis=0)

        if self.view_ratio >0: #测试鲁棒性
            input_image_norm = self.zero_out_first_ratio_layers(input_image_norm, drop_ratio=self.view_ratio)




        # 示例用法
        # 假设 input_image_norm 是一个形状为 (28, 600, 600) 的 numpy 数组

        #input_image_norm = randomly_zero_out_layers(input_image_norm, drop_rate=0.3)

        gt_image_norm = (gt_image - p_low) / (p_high - p_low )
        gt_image_norm = np.clip(gt_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围
        #维度调整
        # gt_image_norm = np.transpose(gt_image_norm, (1,0,2))
        # input_image_norm = np.transpose(input_image_norm, (1, 0, 2))
        # gt_image_norm = np.transpose(gt_image_norm, (1,0,2))
        # input_image_norm = np.transpose(input_image_norm, (1, 0, 2))
        # Step 3: 应用额外的自定义变换（如果有）

        if self.train_transform  is not None:
            input_image_norm, gt_image_norm = self.train_transform(input_image_norm, gt_image_norm)

        if self.val_transform is not None:
            input_image_norm, gt_image_norm = self.val_transform(input_image_norm, gt_image_norm)
            #     input_image_norm = self.input_transform(input_image_norm)
            # if self.gt_transform:
            #
            #     gt_image_norm = self.gt_transform(gt_image_norm)
        # if self.input_transform:
        #     input_image_norm = self.input_transform(input_image_norm)
        # if self.gt_transform:
        #     gt_image_norm = self.gt_transform(gt_image_norm)

        input_image_norm =input_image_norm
        gt_image_norm =gt_image_norm

        #检查input_image_norm和gt_image_norm中是否存在NAN，应该是tensor
        # if torch.isnan(input_image_norm).any():
        #     print("input_image_norm has NAN")
        #     print(img_name)
        # if torch.isnan(gt_image_norm).any():
        #     print("gt_image_norm has NAN")
        #     print(img_name)
        #重新写，我的是numpy，检测数据中是否存在nan
        if np.isnan(input_image_norm).any():
            print("input_image_norm has NAN")
            print(img_name)
        if np.isnan(gt_image_norm).any():
            print("gt_image_norm has NAN")
            print(img_name)


        return input_image_norm, gt_image_norm, img_name, p_low, p_high
    @staticmethod
    def randomly_zero_out_pixels(input_image, drop_rate=0.3):
        """
        随机丢弃（置0）输入数组的像素
        :param input_image: 输入的numpy数组，维度为 (C, H, W)
        :param drop_rate: 每个像素被丢弃（置0）的概率，默认为30%
        :return: 处理后的数组
        """
        mask = np.random.rand(*input_image.shape) >= drop_rate  # 生成与输入形状相同的随机掩码
        output_image = input_image * mask  # 应用掩码
        return output_image

    @staticmethod
    def randomly_zero_out_layers(input_image, drop_rate=0.3):
        """
        随机丢弃（置0）输入数组的层
        :param input_image: 输入的numpy数组，维度为 (28, H, W)
        :param drop_rate: 每一层被丢弃（置0）的概率，默认为30%
        :return: 丢弃后的数组
        """
        output_image = input_image.copy()  # 创建副本，避免修改原始数据
        num_layers = output_image.shape[0]  # 获取层数

        # 遍历每一层
        for layer in range(num_layers):
            if random.random() < drop_rate:
                output_image[layer, :, :] = 0  # 将该层置为0

        return output_image

    @staticmethod
    def zero_out_first_ratio_layers(input_image, drop_ratio=0.3):
        """
        将输入数组的前 drop_ratio 比例的层置0
        :param input_image: 输入的numpy数组，维度为 (num_layers, H, W)
        :param drop_ratio: 需要遮住的层的比例（例如 0.3 表示置0前 30% 的层）
        :return: 处理后的数组
        """
        output_image = input_image.copy()  # 创建副本，避免修改原始数据
        num_layers = output_image.shape[0]  # 获取总层数
        n = int(drop_ratio * num_layers)  # 向下取整得到要遮住的层数

        # 将前 n 个层置0
        for layer in range(n):
            output_image[layer, :, :] = 0

        return output_image



class InferenceTiffDataset(Dataset):
    def __init__(self, txt_file, transform=None, raw=False):
        """
        推理专用的Dataset，仅加载输入图像并进行必要的处理。

        参数:
        - txt_file: 包含推理图像路径的文本文件。
        - transform: 可选的图像变换。
        - raw: 是否进行原始图像特殊处理。
        """
        self.transform = transform
        self.raw = raw

        # 从 txt 文件中读取要加载的文件路径
        with open(txt_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像路径
        input_path = self.image_files[idx]
        _, img_name = os.path.split(input_path)

        # 读取图像
        input_image = tiff.imread(input_path).astype(np.float32)

        # Step 1: 对输入图像进行百分位归一化处理
        if self.raw:
            input_image = input_image - 160
            input_image[input_image < 0] = 0  # 截断负值

        # 设置百分位范围
        p_low = 0
        p_high = 2000
        input_image_norm = (input_image - p_low) / (p_high - p_low)
        input_image_norm = np.clip(input_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围

        if self.raw:
            # 增加一个0维度
            input_image_norm = input_image_norm[np.newaxis, :, :]  # shape (1, H, W)
            # 将第0通道复制三次变成3通道
            input_image_norm = np.repeat(input_image_norm, 3, axis=0)

        # Step 2: 应用额外的变换（如果有）
        if self.transform is not None:
            input_image_norm = self.transform(input_image_norm)

        #添加一个维度
        #input_image_norm = input_image_norm[np.newaxis, :, :,:]  # shape (1, H, W)

        # 返回推理图像及其相关信息
        return input_image_norm, img_name, p_low, p_high



class CustomTiffDatasetSinglePercentile_fft(Dataset):
    #def __init__(self, input_dir, gt_dir, txt_file, input_transform=None, gt_transform=None):
    def __init__(self, input_dir, gt_dir, txt_file, input_transform=None, gt_transform=None,train_transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.train_transform = train_transform
        # 从 txt 文件中读取要加载的文件名
        with open(txt_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image = np.transpose(input_image, (1, 2, 0))  # 转置通道顺序 (H, W, C)

        # 加载对应的标签图像
        gt_path = os.path.join(self.gt_dir, img_name)
        gt_image = tiff.imread(gt_path).astype(np.float32)
        gt_image = np.transpose(gt_image, (1, 2, 0))  # 转置通道顺序 (H, W, C)

        # Step 1: 对输入图像计算百分位并归一化
        # input_image_norm, p_low, p_high = normalize_percentile(input_image, low=0.5, high=99.5)
        p_low = 0
        p_high = 2000
        input_image_norm =(input_image - p_low) / (p_high - p_low )
        input_image_norm = np.clip(input_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围
        # Step 2: 使用输入图像的百分位值对标签图像进行归一化
        #gt_image_norm = (gt_image - p_low) / (p_high - p_low + 1e-3)  # 使用输入图像的百分位

        gt_image_norm = (gt_image - p_low) / (p_high - p_low )
        gt_image_norm = np.clip(gt_image_norm, 0, 1)  # 裁剪到 [0, 1] 范围

        # Step 3: 应用额外的自定义变换（如果有）

        if self.train_transform  is not None:
            input_image_norm, gt_image_norm = self.train_transform(input_image_norm, gt_image_norm)
        else:
            if self.input_transform:

                input_image_norm = self.input_transform(input_image_norm)
            if self.gt_transform:

                gt_image_norm = self.gt_transform(gt_image_norm)
        # if self.input_transform:
        #     input_image_norm = self.input_transform(input_image_norm)
        # if self.gt_transform:
        #     gt_image_norm = self.gt_transform(gt_image_norm)
        torch.fft.fft2()
        return input_image_norm, gt_image_norm, img_name, p_low, p_high




# class CustomTiffDataset_percentile(Dataset):
#     def __init__(self, input_dir, gt_dir, percentiles, input_transform=None, gt_transform=None):
#         self.input_dir = input_dir
#         self.gt_dir = gt_dir
#         self.percentiles = percentiles  # 传入计算好的百分位数
#         self.input_transform = input_transform
#         self.gt_transform = gt_transform
#         self.image_files = os.listdir(input_dir)
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         # 加载输入图像
#         img_name = self.image_files[idx]
#         input_path = os.path.join(self.input_dir, img_name)
#         input_image = tiff.imread(input_path).astype(np.float32)
#         input_image = np.transpose(input_image, (1, 2, 0))  # 转置
#
#         # 加载对应的标签
#         gt_path = os.path.join(self.gt_dir, img_name)
#         gt_image = tiff.imread(gt_path).astype(np.float32)
#         gt_image = np.transpose(gt_image, (1, 2, 0))  # 转置
#
#         # 使用预先计算好的百分位数进行归一化
#         input_low_perc, input_high_perc, gt_low_perc, gt_high_perc = self.percentiles
#         input_image = minmax_percentile_normalize(input_image, input_low_perc, input_high_perc)
#         gt_image = minmax_percentile_normalize(gt_image, gt_low_perc, gt_high_perc)
#
#         if self.input_transform:
#             input_image = self.input_transform(input_image)
#         if self.gt_transform:
#             gt_image = self.gt_transform(gt_image)
#
#         return input_image, gt_image, img_name

def compute_percentiles(dataloader,
                        save_path='/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/fish1_percentiles.npy',
                        low_percentile=0.5,
                        high_percentile=99.5):
    # 检查是否存在已保存的结果
    if os.path.exists(save_path):
        print("Loading percentiles values from file...")
        percentile_values = np.load(save_path)
    else:
        print("Computing and saving percentiles values...")
        input_values = []
        gt_values = []

        # 计算所有输入和标签的数据，用于计算百分位
        for batch in tqdm(dataloader, desc="Data Processing"):
            inputs, gts, _ = batch
            input_values.extend(inputs.flatten().tolist())  # 展平成1D
            gt_values.extend(gts.flatten().tolist())

        # 计算0.5%和99.5%的百分位
        input_low_perc = np.percentile(input_values, low_percentile)
        input_high_perc = np.percentile(input_values, high_percentile)
        gt_low_perc = np.percentile(gt_values, low_percentile)
        gt_high_perc = np.percentile(gt_values, high_percentile)

        percentile_values = np.array([input_low_perc, input_high_perc, gt_low_perc, gt_high_perc])

        # 保存结果到文件
        np.save(save_path, percentile_values)

    return percentile_values


def minmax_percentile_normalize(image, low, high):
    # 根据百分位归一化，并裁剪到 [0, 1] 范围
    normalized_image = (image - low) / (high - low)
    normalized_image = np.clip(normalized_image, 0, 1)
    return normalized_image










class CustomTiffInferenceDataset(Dataset):
    def __init__(self, input_dir, input_transform=None):
        self.input_dir = input_dir
        self.input_transform = input_transform
        self.image_files = os.listdir(input_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image = np.transpose(input_image, (1, 2, 0))

        if self.input_transform:
            input_image = self.input_transform(input_image)

        return input_image, img_name


def compute_min_max(dataloader,
                             save_path='/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/fish1_vadation500.npy'
                    ):
    # 检查是否存在已保存的结果
    if os.path.exists(save_path):
        print("Loading min/max values from file...")
        min_max_values = np.load(save_path)
    else:
        print("Computing and saving min/max values...")
        input_min_val = float('inf')
        input_max_val = float('-inf')
        gt_min_val = float('inf')
        gt_max_val = float('-inf')

        # 计算最小值和最大值
        for batch in tqdm(dataloader, desc="Data Processing"):
            inputs, gts,_ = batch
            input_min_val = min(input_min_val, inputs.min().item())
            input_max_val = max(input_max_val, inputs.max().item())
            gt_min_val = min(gt_min_val, gts.min().item())
            gt_max_val = max(gt_max_val, gts.max().item())

        min_max_values = np.array([input_min_val, input_max_val, gt_min_val, gt_max_val])

        # 保存结果到文件
        np.save(save_path, min_max_values)

    return min_max_values



def compute_mean_std(dataloader, save_path='/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/RLD60_paired1500_mean_std.npy'):
    # 检查是否存在已保存的结果
    if os.path.exists(save_path):
        print("Loading mean/std values from file...")
        mean_std_values = np.load(save_path)
    else:
        print("Computing and saving mean/std values...")
        input_sum, input_sum_sq, gt_sum, gt_sum_sq, count = 0, 0, 0, 0, 0

        # 计算均值和标准差
        for batch in tqdm(dataloader, desc="Data Processing"):
            inputs, labels, _ = batch
            input_sum += torch.sum(inputs)
            input_sum_sq += torch.sum(inputs ** 2)
            gt_sum += torch.sum(labels)
            gt_sum_sq += torch.sum(labels ** 2)
            count += torch.numel(inputs)

        input_mean = input_sum / count
        input_std = torch.sqrt(input_sum_sq / count - input_mean ** 2)
        gt_mean = gt_sum / count
        gt_std = torch.sqrt(gt_sum_sq / count - gt_mean ** 2)

        mean_std_values = np.array([input_mean.item(), input_std.item(), gt_mean.item(), gt_std.item()])

        # 保存结果到文件
        np.save(save_path, mean_std_values)

    return mean_std_values



#多线程版本：
# def compute_min_max(dataloader,
#                     save_path='/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/min_max_values1.npy',
#                     max_workers=4):
#     # 检查是否存在已保存的结果
#     if os.path.exists(save_path):
#         print("Loading min/max values from file...")
#         min_max_values = np.load(save_path)
#     else:
#         print("Computing and saving min/max values...")
#
#         input_min_val = float('inf')
#         input_max_val = float('-inf')
#         gt_min_val = float('inf')
#         gt_max_val = float('-inf')
#
#         def process_batch(batch):
#             inputs, gts, _ = batch
#             return (inputs.min().item(), inputs.max().item(), gts.min().item(), gts.max().item())
#
#         # Create a ThreadPoolExecutor with the specified number of workers
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all batch processing tasks
#             futures = [executor.submit(process_batch, batch) for batch in dataloader]
#
#             # Use tqdm to visualize progress
#             with tqdm(total=len(futures), desc="Data Processing") as pbar:
#                 for future in as_completed(futures):
#                     batch_min_input, batch_max_input, batch_min_gt, batch_max_gt = future.result()
#                     input_min_val = min(input_min_val, batch_min_input)
#                     input_max_val = max(input_max_val, batch_max_input)
#                     gt_min_val = min(gt_min_val, batch_min_gt)
#                     gt_max_val = max(gt_max_val, batch_max_gt)
#                     pbar.update(1)  # Update progress bar for each completed future
#
#         min_max_values = np.array([input_min_val, input_max_val, gt_min_val, gt_max_val])
#
#         # 保存结果到文件
#         np.save(save_path, min_max_values)
#
#     return min_max_values


class MinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        img = (img - self.min_val) / (self.max_val - self.min_val)
        return img


class InverseMinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        img = img * (self.max_val - self.min_val) + self.min_val
        return img



class CustomTiffDataset_double_gt(Dataset):
    def __init__(self, input_dir, gt_dir1, gt_dir2, input_transform=None, gt_transform=None, gt2_transform=None):
        """
        初始化自定义数据集
        input_dir: 输入图像目录
        gt_dir1: 第一个GT图像目录
        gt_dir2: 第二个GT图像目录
        input_transform: 输入图像的变换
        gt_transform: GT图像的变换
        """
        self.input_dir = input_dir
        self.gt_dir1 = gt_dir1
        self.gt_dir2 = gt_dir2
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.image_files = os.listdir(input_dir)
        self.gt2_transform = gt2_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        input_image = np.transpose(input_image, (1, 2, 0))

        # 加载第一个GT图像
        gt_path1 = os.path.join(self.gt_dir1, img_name)
        gt_image1 = tiff.imread(gt_path1).astype(np.float32)
        gt_image1 = np.transpose(gt_image1, (1, 2, 0))

        # 加载第二个GT图像
        gt_path2 = os.path.join(self.gt_dir2, img_name)
        gt_image2 = tiff.imread(gt_path2).astype(np.float32)
        gt_image2 = np.transpose(gt_image2, (1, 2, 0))
        #numpy增加一个0维度
        #gt_image2 =  np.expand_dims(gt_image2, axis=0)
        #gt_image2 = np.transpose(gt_image2, (1, 2, 0))

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.gt_transform:
            gt_image1 = self.gt_transform(gt_image1)
        if self.gt2_transform:
            gt_image2 = self.gt2_transform(gt_image2)

        return input_image, gt_image1, gt_image2, img_name

def compute_min_max_double_gt(dataloader, save_path='/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/double_fish1_1500.npy'):
    # 检查是否存在已保存的结果
    if os.path.exists(save_path):
        print("Loading min/max values from file...")
        min_max_values = np.load(save_path)
    else:
        print("Computing and saving min/max values...")
        input_min_val = float('inf')
        input_max_val = float('-inf')
        gt1_min_val = float('inf')
        gt1_max_val = float('-inf')
        gt2_min_val = float('inf')
        gt2_max_val = float('-inf')

        # 计算最小值和最大值
        for batch in tqdm(dataloader, desc="Data Processing"):
            inputs, gt1, gt2,_ = batch
            input_min_val = min(input_min_val, inputs.min().item())
            input_max_val = max(input_max_val, inputs.max().item())
            gt1_min_val = min(gt1_min_val, gt1.min().item())
            gt1_max_val = max(gt1_max_val, gt1.max().item())
            gt2_min_val = min(gt2_min_val, gt2.min().item())
            gt2_max_val = max(gt2_max_val, gt2.max().item())

        min_max_values = np.array([input_min_val, input_max_val, gt1_min_val, gt1_max_val, gt2_min_val, gt2_max_val])

        # 保存结果到文件
        np.save(save_path, min_max_values)
    return min_max_values


class CustomTiffDataset_FP(Dataset):
    def __init__(self, input_dir, gt_dir, file_list, transform=None):
        """
        Dataset for loading input and ground truth tiff images based on file_list with optional transformations.

        :param input_dir: Directory containing input images.
        :param gt_dir: Directory containing ground truth images.
        :param file_list: List of file names (from train.txt or val.txt).
        :param transform: A transformation (augmentation or normalization) function to be applied on both input and ground truth.
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.file_list = file_list  # List of file names from txt
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image file name from the list
        img_name = self.file_list[idx]

        # Load input image
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)

        # Load ground truth image
        gt_path = os.path.join(self.gt_dir, img_name)
        gt_image = tiff.imread(gt_path).astype(np.float32)

        # Add an extra dimension to gt_image if needed
        #gt_image = np.expand_dims(gt_image, axis=0)


        #都增加一个0维度




        # Apply the transformation (could be augmentation or normalization)
        if self.transform:
            input_image, gt_image = self.transform(input_image, gt_image)

        return input_image, gt_image, img_name

class ValTransform:
    def __init__(self, input_min_val, input_max_val, gt_min_val, gt_max_val):
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val

    def __call__(self, input_img, gt_img):
        # Convert to tensor and normalize
        input_img = F.to_tensor(input_img)
        input_img = F.normalize(input_img, [self.input_min_val], [self.input_max_val])
        input_img = input_img.permute(1, 0, 2)


        gt_img = F.to_tensor(gt_img)
        gt_img = F.normalize(gt_img, [self.gt_min_val], [self.gt_max_val])
        gt_img = gt_img.permute(1, 0, 2)

        return input_img, gt_img

class SynchronizedTransform:
    def __init__(self, input_min_val, input_max_val, gt_min_val, gt_max_val):
        """
        A class to apply synchronized transformations to input and ground truth images.
        :param input_min_val: Minimum value for input image normalization.
        :param input_max_val: Maximum value for input image normalization.
        :param gt_min_val: Minimum value for ground truth normalization.
        :param gt_max_val: Maximum value for ground truth normalization.
        """
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val

    def __call__(self, input_img, gt_img):
        # Random horizontal flip
        input_img = F.to_tensor(input_img)
        gt_img = F.to_tensor(gt_img)

        #input_img维度调整
        input_img = input_img.permute(1, 0, 2)
        gt_img = gt_img.permute(1, 0, 2)



        if random.random() > 0.5:
            input_img = F.hflip(input_img)
            gt_img = F.hflip(gt_img)

        # Random vertical flip
        if random.random() > 0.5:
            input_img = F.vflip(input_img)
            gt_img = F.vflip(gt_img)

        # Random rotation
        angle = random.uniform(-30, 30)
        input_img = F.rotate(input_img, angle)
        gt_img = F.rotate(gt_img, angle)

        # Convert to tensor and normalize
        #input_img = F.to_tensor(input_img)
        input_img = F.normalize(input_img, [self.input_min_val], [self.input_max_val])

        #gt_img = F.to_tensor(gt_img)
        gt_img = F.normalize(gt_img, [self.gt_min_val], [self.gt_max_val])




        return input_img, gt_img




class CustomTiffDataset_FP_Inference(Dataset):
    def __init__(self, input_dir,  input_transform=None):
        self.input_dir = input_dir
        self.input_transform = input_transform
        self.image_files = os.listdir(input_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载输入图像
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        input_image = tiff.imread(input_path).astype(np.float32)
        #input_image增加一个0维度
        #input_image = np.expand_dims(input_image, axis=0)
        # 加载对应的标签

        #gt_image = np.expand_dims(gt_image, axis=0)
        #gt_image = np.transpose(gt_image,(1,2,0))
        if self.input_transform:
            input_image = self.input_transform(input_image)
            input_image = input_image.permute(1, 0, 2)
            input_image = input_image.squeeze(0)

        #tensor的维度修改




        return input_image,img_name






if __name__ == "__main__":
#主函数，用于进行调试
# 创建归一化和逆归一化实例
#测试CustomTiffDataset_double_gt
    input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/input1500_location'
    gt_dir1 = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD60_1500'
    gt_dir2 = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD60_1500_conv'
    input_transform = transforms.Compose([transforms.ToTensor()])
    gt_transform = transforms.Compose([transforms.ToTensor()])
    gt2_transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomTiffDataset_double_gt(input_dir, gt_dir1, gt_dir2, input_transform, gt_transform, gt2_transform)
    DataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    #调用dataloader
    min_max_values = compute_min_max_double_gt(DataLoader)
    print(min_max_values)
    for batch in DataLoader:
        inputs, gt1, gt2 = batch
        print(inputs.shape, gt1.shape, gt2.shape)
        break

