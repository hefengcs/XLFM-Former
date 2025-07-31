import numpy as np
import tifffile
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def calculate_mse(tif_path1, tif_path2):
    """
    计算两个归一化的 TIFF 文件之间的均方误差 (MSE)。

    参数:
    tif_path1 (str): 第一个 TIFF 文件的路径。
    tif_path2 (str): 第二个 TIFF 文件的路径。

    返回:
    float: 两幅归一化图像之间的均方误差。
    """
    # 读取 TIFF 文件
    image1 = tifffile.imread(tif_path1).astype(np.float32)
    image2 = tifffile.imread(tif_path2).astype(np.float32)

    # 确保图像具有相同的尺寸
    if image1.shape != image2.shape:
        return None  # 返回 None 表示文件不匹配或有问题

    # 归一化图像到 [0, 1] 范围
    image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    # 计算均方误差
    mse = np.mean((image1_normalized - image2_normalized) ** 2)
    return mse


def calculate_mse_and_diff(folder1, folder2, output_csv):
    """
    计算两个文件夹中所有对应 TIFF 文件的 MSE 及其差值，并保存到 CSV 文件。

    参数:
    folder1 (str): 第一个文件夹的路径。
    folder2 (str): 第二个文件夹的路径。
    output_csv (str): 保存结果的 CSV 文件路径。
    """
    files1 = sorted(os.listdir(folder1))  # 按文件名排序
    mse_values = []
    future_to_file = {}

    with ThreadPoolExecutor(max_workers=32) as executor:
        with tqdm(total=len(files1), desc="Processing Files") as pbar:
            for file1 in files1:
                if file1.endswith('.tif'):
                    path1 = os.path.join(folder1, file1)
                    path2 = os.path.join(folder2, file1)  # 假设两个文件夹中的文件名对应
                    if os.path.exists(path2):
                        future = executor.submit(calculate_mse, path1, path2)
                        future_to_file[future] = file1

            # Collect results as they complete
            for future in as_completed(future_to_file):
                mse = future.result()
                if mse is not None:
                    mse_values.append((future_to_file[future], mse))  # 保存文件名和对应的MSE
                pbar.update(1)

    # 按文件名排序
    mse_values.sort(key=lambda x: x[0])

    # 计算 MSE 差值
    mse_diffs = [0]  # 第一个文件没有前一个文件进行比较，差值设为0
    for i in range(1, len(mse_values)):
        mse_diffs.append(mse_values[i][1] - mse_values[i - 1][1])

    # 将结果写入 CSV 文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File', 'MSE', 'MSE_Diff'])  # 写入表头
        for i, (filename, mse) in enumerate(mse_values):
            writer.writerow([filename, mse, mse_diffs[i]])


# 示例使用
output_csv_path = '/home/LifeSci/wenlab/hefengcs/VCD_dataset/moving_fish2/mse_results.csv'
calculate_mse_and_diff('/home/LifeSci/wenlab/hefengcs/VCD_dataset/moving_fish2/gt4000_RLD27',
                       '/home/LifeSci/wenlab/hefengcs/VCD_dataset/moving_fish2/fish2_gt_RLD60',
                       output_csv_path)
print(f"结果已保存到: {output_csv_path}")
