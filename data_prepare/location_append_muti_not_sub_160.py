import tifffile as tiff
from scipy.ndimage import zoom
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 定义输入目录和输出目录
input_dir_1 = 'anonymousVCD_dataset/fixed_fish/240903-2/input_location_clean'
input_dir_2 = 'anonymousVCD_dataset/fixed_fish/240903-2/clean_g'
output_dir = input_dir_1

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取两个输入目录中的文件名集合
files_1 = set(os.listdir(input_dir_1))
files_2 = set(os.listdir(input_dir_2))

# 找到两个目录中相同的文件名
common_files = files_1.intersection(files_2)

# 定义处理单个文件的函数
def process_file(filename):
    try:
        # 构建完整的文件路径
        path_1 = os.path.join(input_dir_1, filename)
        path_2 = os.path.join(input_dir_2, filename)

        # 读取图像
        image = tiff.imread(path_1)
        image_2 = tiff.imread(path_2)

        # 计算缩放因子，将2048x2048的图像调整为600x600
        zoom_factors = (600 / image_2.shape[0], 600 / image_2.shape[1])
        image_2_resized = zoom(image_2, zoom_factors, order=1)

        # 转换为int32类型以处理减法操作
        image_2_resized = image_2_resized.astype(np.int32)

        # 执行减法操作，并确保结果不会小于0
        #image_2_resized = np.maximum(image_2_resized - 160, 0)

        # 确保数值不超出uint16范围，并转换回uint16
        image_2_resized = np.clip(image_2_resized, 0, 65535).astype(np.uint16)

        # 将调整后的图像添加到27个切片的图像数据上
        image_2_resized = image_2_resized[np.newaxis, :, :]  # shape (1, 600, 600)

        # 确保原始图像也为uint16
        image = image.astype(np.uint16)

        # 使用np.concatenate堆叠
        combined_image = np.concatenate((image, image_2_resized), axis=0)

        # 保存为新的tif文件
        output_path = os.path.join(output_dir, filename)
        tiff.imwrite(output_path, combined_image)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 使用ThreadPoolExecutor来并行处理
with ThreadPoolExecutor(max_workers=32) as executor:
    list(tqdm(executor.map(process_file, common_files), total=len(common_files)))

print("Batch processing completed.")
