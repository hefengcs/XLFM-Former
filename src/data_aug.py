import numpy as np
import random
from scipy.ndimage import rotate
import torchvision.transforms as transforms

class RandomHorizontalFlip:
    """随机水平翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, seed=None):
        # 如果传入种子，则使用相同的种子
        if seed is not None:
            random.seed(seed)

        # 判断是否进行水平翻转
        if random.random() < self.p:
            return np.flip(image, axis=1).copy()  # axis=1 是水平翻转
        return image


class RandomVerticalFlip:
    """随机垂直翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, seed=None):
        # 如果传入种子，则使用相同的种子
        if seed is not None:
            random.seed(seed)

        # 判断是否进行垂直翻转
        if random.random() < self.p:
            return np.flip(image, axis=0).copy()  # axis=0 是垂直翻转
        return image


class RandomRotation:
    """随机旋转，只旋转90度的倍数"""

    def __init__(self, angles=(0, 90, 180, 270), p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, image, seed=None):
        # 如果传入种子，则使用相同的种子

        if seed is not None:
            random.seed(seed)

        # 从给定的角度列表中随机选择一个角度
        angle = random.choice(self.angles)

        # 旋转每张图像，保留第三维度
        rotated_images = np.empty_like(image)
        for i in range(image.shape[2]):
            # 旋转单张图像
            rotated_images[:, :, i] = rotate(image[:, :, i], angle, reshape=False, mode='nearest')

        if random.random() < self.p:
            return rotated_images
        return image


class ToTensor:
    """将 NumPy 数组转换为 Tensor（兼容 PyTorch 风格）"""

    def __call__(self, image, seed=None):
        # 转换为 PyTorch 风格的张量 (C, H, W)
        #image = image.transpose((2, 0, 1)).astype(np.float32)

        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            # MinMaxNormalize(val_input_min, val_input_max)
        ])
        image = tensor_transform(image)
        return  image  # 将 (H, W, C) 转为 (C, H, W)



class Compose:
    """组合多个数据增强操作，根据需要分别对输入图像和mask进行增强"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        seed = random.randint(0, 10000)  # 生成一个随机种子

        for t in self.transforms:
            # 针对高斯噪声和高斯模糊，只对输入图像进行增强
            if isinstance(t, (RandomGaussianNoise, RandomGaussianBlur)):
                image = t(image, seed=seed)
            else:
                # 如果提供了 mask，则对 image 和 mask 同时进行增强
                if mask is not None:
                    image = t(image, seed=seed)
                    mask = t(mask, seed=seed)
                else:
                    # 如果没有提供 mask，则只对 image 进行增强
                    image = t(image, seed=seed)

        if mask is not None:
            return image, mask
        return image

class RandomBrightness:
    """随机亮度调整"""

    def __init__(self, delta=0.2, p=0.5):
        """
        :param delta: 最大亮度调整范围 (-delta, delta)
        :param p: 应用调整的概率
        """
        self.delta = delta
        self.p = p

    def __call__(self, image, seed=None):
        if seed is not None:
            random.seed(seed)

        if random.random() < self.p:
            # 计算调整因子
            factor = 1 + random.uniform(-self.delta, self.delta)
            return np.clip(image * factor, 0, 1)  # 假设图像数据在[0, 1]范围内
        return image

class RandomContrast:
    """随机对比度调整"""

    def __init__(self, factor=0.2, p=0.5):
        """
        :param factor: 对比度调整因子 (1-factor, 1+factor)
        :param p: 应用调整的概率
        """
        self.factor = factor
        self.p = p

    def __call__(self, image, seed=None):
        if seed is not None:
            random.seed(seed)

        if random.random() < self.p:
            mean = np.mean(image)
            # 应用对比度调整
            return np.clip((1 + random.uniform(-self.factor, self.factor)) * (image - mean) + mean, 0, 1)
        return image


class RandomGaussianNoise:
    """随机高斯噪声"""

    def __init__(self, mean=0.0, std=0.1, p=0.5):
        """
        :param mean: 高斯噪声的均值
        :param std: 高斯噪声的标准差
        :param p: 应用噪声的概率
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)  # 确保 NumPy 的随机数生成器也使用相同的种子

        if random.random() < self.p:
            noise = np.random.normal(self.mean, self.std, image.shape)
            return np.clip(image + noise, 0, 1)  # 假设图像数据在[0, 1]范围内
        return image

import cv2

class RandomGaussianBlur:
    """随机高斯模糊"""

    def __init__(self, kernel_size=(3, 3), sigma_range=(0.1, 2.0), p=0.5):
        """
        :param kernel_size: 高斯模糊的核大小，必须是奇数 (height, width)
        :param sigma_range: 模糊的标准差范围
        :param p: 应用模糊的概率
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, seed=None):
        if seed is not None:
            random.seed(seed)

        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            # 对每个通道分别应用高斯模糊
            blurred_images = np.empty_like(image)
            for i in range(image.shape[2]):  # 针对每张灰度图进行模糊
                blurred_images[:, :, i] = cv2.GaussianBlur(image[:, :, i], self.kernel_size, sigma)
            return blurred_images
        return image



#定义主函数用于测试
if __name__ == '__main__':
    # 创建输入 NumPy 数组，形状为 (600, 600, 28)
    input_image = np.random.rand(600, 600, 28) * 255
    input_image = input_image.astype(np.uint8)

    # 创建相应的标签图像（假设是二值化的 mask），形状相同
    mask_image = np.random.randint(0, 2, (600, 600, 300)) * 255
    mask_image = mask_image.astype(np.uint8)

    # 定义增强操作序列，类似于 torchvision.transforms.Compose
    train_transform = Compose([
        #RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        RandomVerticalFlip(p=0.5),  # 50% 概率垂直翻转
        RandomRotation(p=0.5),  # 只旋转
        ToTensor(),  # 转换为 Tensor 形式
    ])

    # 应用数据增强
    augmented_input_image, augmented_mask_image = train_transform(input_image, mask_image)

    # augmented_input_image 和 augmented_mask_image 使用相同的增强操作
    print(augmented_input_image.shape)  # 输出: (28, 600, 600)
    print(augmented_mask_image.shape)  # 输出: (28, 600, 600)
