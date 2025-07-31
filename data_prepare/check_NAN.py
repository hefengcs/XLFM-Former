import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# 输入文件夹路径
input_folder = "anonymousVCD_dataset/moving_fish4/input_location"  # 替换为你的 TIFF 文件目录路径

def check_nan_in_tiff(file_path):
    """
    检查给定的 TIFF 文件是否包含 NaN。
    :param file_path: TIFF 文件的路径
    :return: 如果包含 NaN 返回 True，否则返回 False
    """
    try:
        # 读取 TIFF 文件
        image = tiff.imread(file_path)

        # 检查是否包含 NaN
        if np.isnan(image).any():
            return True
    except Exception as e:
        # 如果 TIFF 文件无法读取，打印错误
        print(f"Error reading file {file_path}: {e}")
    return False

# 获取所有 TIFF 文件的路径
tiff_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

# 遍历所有 TIFF 文件并检查 NaN
print("Checking for NaN values in TIFF files...")
for file_path in tqdm(tiff_files, desc="Checking TIFF files"):
    if check_nan_in_tiff(file_path):
        print(f"NaN found in file: {file_path}")
