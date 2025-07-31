import os
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 输入文件夹路径
input_folder = "anonymousVCD_dataset/moving_fish4/input_location_case3"  # 替换为你的输入文件夹路径

# 获取文件夹中的所有 TIFF 文件
def get_tiff_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.tiff') or f.endswith('.tif')]

# 处理单个文件：计算最大值并检查是否小于阈值
def process_file(file_name, limit):
    file_path = os.path.join(input_folder, file_name)

    # 读取 TIFF 文件
    image = tiff.imread(file_path)

    # 计算最大值
    max_value = image.max()

    # 返回符合条件的文件名
    if max_value < limit:
        return file_name
    return None

# 遍历文件夹中的文件，获取最大值小于阈值的文件名
def find_files_with_max_less_than(limit):
    files = get_tiff_files(input_folder)
    matching_files = []

    # 使用多线程处理
    with ThreadPoolExecutor() as executor:
        # tqdm 进度条
        results = list(tqdm(executor.map(lambda f: process_file(f, limit), files), total=len(files), desc="Processing files"))

    # 收集非空结果
    matching_files = [file for file in results if file is not None]
    return matching_files

# 设置阈值
threshold = 800

# 获取结果
files_with_low_max = find_files_with_max_less_than(threshold)

# 输出结果
print("Files with max value less than 900:")
for file_name in files_with_low_max:
    print(file_name)
