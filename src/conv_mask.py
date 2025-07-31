import os
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

def normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def ifftshift(x):
    shift = [-(dim // 2) for dim in x.shape]
    return torch.roll(x, shift, dims=(-2, -1))

def pad_and_assign(matrix, target_size):
    padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=matrix.device)
    padded_matrix[:, 852:1452, 852:1452] = matrix
    return padded_matrix
def pad_to_target(matrix, target_size):
    padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=matrix.device)
    z_start = (target_size[0] - matrix.size(0)) // 2
    y_start = (target_size[1] - matrix.size(1)) // 2
    x_start = (target_size[2] - matrix.size(2)) // 2

    padded_matrix[z_start:z_start + matrix.size(0),
    y_start:y_start + matrix.size(1),
    x_start:x_start + matrix.size(2)] = matrix
    return padded_matrix

# 读取 TIFF 文件
def read_tiff(tiff_path):
    matrix = tiff.imread(tiff_path)
    matrix = matrix.astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32, device=device)


# 读取 HDF5 文件
def read_hdf5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        matrix = f['PSF_1'][:]
        matrix = matrix.astype(np.float32)
        tiff.imwrite('/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/PSF_G.tif', matrix)
        matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
        matrix = matrix.permute(0, 2, 1)  # 调整维度顺序
    return matrix


# 从txt文件读取27个坐标
def read_coordinates(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        coordinates = [list(map(int, line.split()[:2])) for line in lines]
    return coordinates


def create_circular_mask(image_size, center, radius):
    mask = np.zeros(image_size, dtype=np.uint8)
    y0, x0 = center  # 圆心的坐标
    Y, X = np.ogrid[:image_size[0], :image_size[1]]  # 生成坐标网格

    # 计算 (x - x0)^2 + (y - y0)^2 <= r^2 的区域
    dist_from_center = (X - x0) ** 2 + (Y - y0) ** 2
    mask[dist_from_center <= radius ** 2] = 1  # 半径内的点设置为1

    return torch.tensor(mask, dtype=torch.float32, device=device)


# 逐步生成 mask 和处理傅里叶卷积
def process_mask_and_conv(matrix1, matrix2, coord):
    x, y = coord
    image_size = (2048, 2048)
    target_size = (300, 2304, 2304)

    # 生成300x300的mask
    # mask = np.zeros(image_size, dtype=np.uint8)
    # mask[y:y + 200, x:x + 200] = 1
    # mask = torch.tensor(mask, dtype=torch.float32, device=device)
    radius = 300  # 半径
    mask = create_circular_mask(image_size, (x, y), radius)





    # 应用 mask 到 PSF 矩阵
    masked_psf = matrix2 * mask
    padded_matrix1 = pad_and_assign(matrix1, target_size)
    padded_matrix2 = pad_to_target(masked_psf, target_size)
    # 进行傅里叶卷积
    #convolved_slice = fourier_conv(padded_matrix1, masked_psf)

    summed_matrix = torch.zeros(padded_matrix1.size(1), padded_matrix1.size(2), dtype=torch.float32, device=device)

    for i in range(padded_matrix1.size(0)):
        convolved_slice = fourier_conv(padded_matrix1[i], padded_matrix2[i])
        summed_matrix += convolved_slice

    # 裁剪出2048x2048的区域
    cropped_matrix = summed_matrix[128:-128, 128:-128]


    # 裁剪傅里叶卷积结果出 600x600 的区域
    cropped = cropped_matrix[y:y + 600, x:x + 600]

    return cropped


# 进行傅里叶卷积
def fourier_conv(image, psf):
    psf = ifftshift(psf)
    fft_image = torch.fft.fft2(image)
    fft_psf = torch.fft.fft2(psf)
    result = torch.real(torch.fft.ifft2(fft_image * fft_psf))
    return result


# 处理文件
def process_file(tiff_path, matrix2, output_dir, coordinates):
    matrix1 = read_tiff(tiff_path)
    # 归一化
    matrix1 = normalize(matrix1)
    matrix2 = normalize(matrix2)
    # 初始化存储裁剪结果的列表
    cropped_matrices = []

    # 对每个坐标执行卷积并裁剪
    for coord in coordinates:


        cropped = process_mask_and_conv(matrix1, matrix2, coord)
        #不足600，600的，padding0到600，600
        if cropped.size(0) < 600 or cropped.size(1) < 600:
            padded = torch.zeros(600, 600, dtype=torch.float32, device=device)
            padded[:cropped.size(0), :cropped.size(1)] = cropped
            cropped = padded

        cropped_matrices.append(cropped)

    # 将27个600x600的裁剪矩阵堆叠
    cropped_matrices = torch.stack(cropped_matrices)

    # 保存最终的裁剪结果
    cropped_matrix_uint16 = np.clip(cropped_matrices.cpu().numpy(), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    filename = os.path.basename(tiff_path)
    output_tiff_path = os.path.join(output_dir, filename)
    tiff.imwrite(output_tiff_path, cropped_matrix_uint16)


# 处理整个目录
def process_directory(tiff_dir, h5_path, output_dir, coordinates_txt):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取HDF5文件
    matrix2 = read_hdf5(h5_path)

    # 读取坐标信息
    coordinates = read_coordinates(coordinates_txt)

    # 处理每个TIFF文件
    for filename in tqdm(os.listdir(tiff_dir), desc="Processing files"):
        if filename.endswith('.tif'):
            tiff_path = os.path.join(tiff_dir, filename)
            process_file(tiff_path, matrix2, output_dir, coordinates)


# 输入路径
tiff_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD60_1500'
output_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/clean_view_1500'
h5_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/RLD_test/PSF_G.mat'
coordinates_txt = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data_prepare/coordinates.txt'

# 处理目录中的所有文件
process_directory(tiff_dir, h5_path, output_dir, coordinates_txt)
