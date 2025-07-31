import os
import tifffile as tiff
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置路径
input_dir = '/gpfsanonymousVCD_dataset/fixed_fish/240725_03/g'
output_dir = '/gpfsanonymousVCD_dataset/fixed_fish/240725_03/flow'

# 检查输出目录是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取目录中的所有tif文件，并按文件名排序 (假设文件名表示时间顺序)
tif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])


# 定义光流计算和保存函数
def compute_and_save_optical_flow(i, img1_path, img2_path):
    # 读取连续两帧图像
    img1 = tiff.imread(img1_path)
    img2 = tiff.imread(img2_path)

    # 如果是三维图像，则只取第一个通道（假设是灰度图）
    if img1.ndim == 3:
        img1 = img1[..., 0]
    if img2.ndim == 3:
        img2 = img2[..., 0]

    # 将图像转换为8位灰度图
    img1_gray = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_gray = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 计算光流（使用Farneback算法）
    flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 将光流的两部分分开 (水平和垂直方向)
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # 保存光流结果为tif文件
    output_file_x = os.path.join(output_dir, f'flow_x_{i:04d}.tif')
    output_file_y = os.path.join(output_dir, f'flow_y_{i:04d}.tif')

    tiff.imwrite(output_file_x, flow_x.astype(np.float32))
    tiff.imwrite(output_file_y, flow_y.astype(np.float32))

    # 返回任务完成标志
    return i


# 多线程处理光流计算
def process_optical_flow_multithread():
    # 创建线程池执行器
    with ThreadPoolExecutor(max_workers=4) as executor:  # 设定4个线程，视硬件情况调整
        futures = []

        # 提交每对图像的光流计算任务
        for i in range(len(tif_files) - 1):
            img1_path = os.path.join(input_dir, tif_files[i])
            img2_path = os.path.join(input_dir, tif_files[i + 1])
            futures.append(executor.submit(compute_and_save_optical_flow, i, img1_path, img2_path))

        # 使用tqdm显示任务进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Optical Flow"):
            future.result()  # 获取每个任务的结果


# 开始多线程光流处理
process_optical_flow_multithread()

print(f"Optical flow calculations completed and saved to {output_dir}")
