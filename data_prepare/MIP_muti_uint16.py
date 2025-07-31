import numpy as np
import imageio
import nibabel as nib
import os
import scipy.io
import tifffile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def image_write(file_obj_path, obj_name, obj_recon, pattern):
    # 将数据从 [-1, 1] 转换为 [0, 65535] 的 uint16 格式


    obj_recon[obj_recon < 1] = 0
    #
    obj_recon_uint16 = obj_recon.astype(np.uint16)

    # 保存重建对象
    if pattern == 1:
        scipy.io.savemat(os.path.join(file_obj_path, obj_name), {'ObjRecon': obj_recon_uint16})
    else:
        # 将 NumPy 数组保存为 NIfTI 文件
        nifti_img = nib.Nifti1Image(obj_recon, np.eye(4))
        nib.save(nifti_img, os.path.join(file_obj_path, obj_name + '.nii'))

def process_single_tiff(tiff_file, tiff_directory, file_obj_path, pattern):
    # 读取单个 TIFF 文件
    obj_recon = tifffile.imread(os.path.join(tiff_directory, tiff_file))

    # 将 2D 图像扩展为 3D（增加一个维度）
    obj_recon = obj_recon.transpose(1, 2, 0)

    # 替换文件名前缀 'VCD' 为 'Green_Recon'
    # obj_name = tiff_file.replace('0', 'Green_Recon_').replace('.tif', '.mat')
    obj_name = 'Green_Recon_' + tiff_file.replace('.tif', '.mat')
    #对文件名添加一个前缀"Green_Recon_",文件名的修改




    # 调用 image_write 函数处理每个 TIF 文件
    image_write(file_obj_path, obj_name, obj_recon, pattern)

def process_tiff_files(tiff_directory, file_obj_path, pattern=1, num_workers=32):
    # 获取目录中所有的 TIF 文件
    tiff_files = sorted([f for f in os.listdir(tiff_directory) if f.endswith('.tif')])

    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in the specified directory.")

    # 使用 tqdm 添加进度条，并使用 ThreadPoolExecutor 来并行处理文件
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda tiff_file: process_single_tiff(tiff_file, tiff_directory, file_obj_path, pattern),
                               tiff_files), total=len(tiff_files), desc="Processing TIFF files"))

# 示例用法
tiff_directory = 'anonymousVCD_torch_gnode05/sample/11_18_test'
file_obj_path = 'anonymousVCD_torch_gnode05/sample/11_18_test/Green_Recon'
pattern = 1  # 保存为 .mat 文件

# 确保输出路径存在
os.makedirs(file_obj_path, exist_ok=True)

# 处理目录中的所有 TIF 文件，使用 32 个线程
process_tiff_files(tiff_directory, file_obj_path, pattern, num_workers=8)
