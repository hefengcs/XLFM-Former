import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn.functional as F
import tifffile as tiff
import h5py
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义 fftshift 和 ifftshift 函数
def fftshift(x):
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, dims=(-2, -1))


def ifftshift(x):
    shift = [-(dim // 2) for dim in x.shape]
    return torch.roll(x, shift, dims=(-2, -1))


# 读取 TIFF 文件
def read_tiff(tiff_path):
    matrix = tiff.imread(tiff_path)
    #转float32
    matrix = matrix.astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32, device=device)


# 读取 HDF5 文件
def read_hdf5(h5_path):
    #with h5py.File(h5_path, 'r') as f:
        #matrix = f['PSF_1'][:]
        matrix =tiff.imread(h5_path)
        # 从uint16转换为float32
        matrix = matrix.astype(np.float32)

        #创建mask
        #image_size = (2048, 2048)

        # 矩形的 x, y 起始位置，宽度和高度
        #x, y, width, height = 463, 0, 300, 300

        # 创建一个空的 mask，初始化为 0
        #mask = np.zeros(image_size, dtype=np.uint8)

        # 将指定矩形区域的值设为 1
        #mask[y:y + height, x:x + width] = 1

        #matrix = matrix * mask

        matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
        matrix = matrix.permute(0, 2, 1)
        return matrix


# 对输入的部分进行归一化
def normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)
    #return tensor


# 确保目标尺寸为 (300, 2304, 2304)
target_size = (300, 2304, 2304)


# 补零操作并确保在两侧补零
def pad_to_target(matrix, target_size):
    padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=matrix.device)
    z_start = (target_size[0] - matrix.size(0)) // 2
    y_start = (target_size[1] - matrix.size(1)) // 2
    x_start = (target_size[2] - matrix.size(2)) // 2

    padded_matrix[z_start:z_start + matrix.size(0),
    y_start:y_start + matrix.size(1),
    x_start:x_start + matrix.size(2)] = matrix
    return padded_matrix


# 重新定义并填充零矩阵，并将600直接赋值给新的张量的第852到1452行和列
def pad_and_assign(matrix, target_size):
    padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=matrix.device)
    padded_matrix[:, 852:1452, 852:1452] = matrix
    return padded_matrix


# 使用傅里叶变换进行卷积
def fourier_conv(image, psf):
    # 中心化 PSF
    psf = ifftshift(psf)
    fft_image = torch.fft.fft2(image)
    fft_psf = torch.fft.fft2(psf)
    result = torch.real(torch.fft.ifft2(fft_image * fft_psf))
    return result


def process_file(tiff_path, matrix2, output_dir):
    # 读取数据
    matrix1 = read_tiff(tiff_path)

    # 归一化
    matrix1 = normalize(matrix1)

    # 补零
    padded_matrix1 = pad_and_assign(matrix1, target_size)
    padded_matrix2 = pad_to_target(matrix2, target_size)

    # 转置为适合傅里叶卷积的格式 (channels, height, width)
    padded_matrix1 = padded_matrix1.permute(0, 1, 2)  # (300, 2304, 2304)
    padded_matrix2 = padded_matrix2.permute(0, 1, 2)  # (300, 2304, 2304)

    # 初始化累加结果矩阵
    summed_matrix = torch.zeros(padded_matrix1.size(1), padded_matrix1.size(2), dtype=torch.float32, device=device)

    # 进行傅里叶卷积并累加
    for i in range(padded_matrix1.size(0)):
        convolved_slice = fourier_conv(padded_matrix1[i], padded_matrix2[i])
        summed_matrix += convolved_slice

    # 裁剪出2048x2048的区域
    cropped_matrix = summed_matrix[128:-128, 128:-128]

    # 将结果转换为uint16
    cropped_matrix = cropped_matrix.cpu().numpy()
    #cropped_matrix_uint16 = np.clip(cropped_matrix, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    # 输出文件名
    #filename = 'PSF_conv'+os.path.basename(tiff_path)
    filename =  os.path.basename(tiff_path)
    output_tiff_path = os.path.join(output_dir, filename)

    # 保存为TIFF文件
    #tiff.imwrite(output_tiff_path, cropped_matrix_uint16)
    tiff.imwrite(output_tiff_path, cropped_matrix)
    #print(f"Cropped convolved matrix saved to {output_tiff_path}")


# 主函数，处理目录中的所有文件
def process_directory(tiff_dir, h5_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取HDF5文件
    matrix2 = read_hdf5(h5_path)
    matrix2 = normalize(matrix2)

    for filename in tqdm(os.listdir(tiff_dir), desc="Processing files"):
        if filename.endswith('.tif'):

            tiff_path = os.path.join(tiff_dir, filename)
            process_file(tiff_path, matrix2, output_dir)


# 输入和输出目录路径
tiff_dir = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/f7/RLD60'
output_dir = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/f7/g_clean'
h5_path = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_ideal/combined_PSF.tif'

# 处理目录中的所有文件
process_directory(tiff_dir, h5_path, output_dir)
