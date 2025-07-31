import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
def check_nan_in_tiff(folder_path):
    """
    检查指定文件夹中的所有 .tif 文件是否包含 NaN 值。

    Args:
        folder_path (str): 文件夹路径

    Returns:
        None
    """
    # 遍历文件夹中的所有文件
    for file_name in tqdm(os.listdir(folder_path)):
        # 检查是否是 .tif 文件
        if file_name.endswith(".tif"):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 使用 tifffile 打开文件
                with tiff.TiffFile(file_path) as tif:
                    # 读取图像数据
                    image_data = tif.asarray()
                    # 检查是否包含 NaN
                    if np.isnan(image_data).any():
                        print(f"File {file_name} contains NaN values.")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# 指定文件夹路径
folder_path = "/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/240903-2/input_location_clean"
check_nan_in_tiff(folder_path)
