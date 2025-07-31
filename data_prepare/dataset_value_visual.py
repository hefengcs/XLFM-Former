# 修改后的代码，取99.5百分位而非最大值
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 定义文件夹路径
tif_folder = 'anonymousVCD_dataset/moving_fish5/input_location'

# 获取文件夹中的所有tif文件
tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])

# 初始化存储每个tif文件99.5百分位值的列表
percentile_values = [0] * len(tif_files)

# 定义处理单个文件的函数，计算99.5百分位
def process_file(idx, tif_file):
    file_path = os.path.join(tif_folder, tif_file)
    image = tiff.imread(file_path)
    percentile_values[idx] = np.percentile(image, 95)

# 使用多线程来处理tif文件
with ThreadPoolExecutor(max_workers=32) as executor:
    list(tqdm(executor.map(lambda idx_tif: process_file(*idx_tif), enumerate(tif_files)), total=len(tif_files)))

# 创建横轴序列
x = list(range(len(percentile_values)))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(x, percentile_values, marker='o', linestyle='-', markersize=2)

# 设置横轴从0到文件数量，且每500个文件标记一次
plt.xticks(np.arange(0, len(percentile_values), 5000))

# 设置纵轴范围从0到2000
plt.ylim(0, 2000)

# 添加标题和标签
plt.title('95th Percentile Value of Each TIFF File')
plt.xlabel('File Index')
plt.ylabel('95th Percentile Value')

# 显示图像
plt.grid(True)
plt.show()
