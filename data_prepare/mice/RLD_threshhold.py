import os
import tifffile
import numpy as np
from tqdm import tqdm
import concurrent.futures

# 设置路径
input_dir = "anonymousVCD_dataset/mice/mice2/subset/RLD60"
output_dir = os.path.join(input_dir, "masked_gt")
os.makedirs(output_dir, exist_ok=True)

# 定义处理函数
def process_file(filename):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not filename.endswith(".tif"):
        return

    volume = tifffile.imread(input_path).astype(np.float32)
    volume[volume < 30] = 0
    tifffile.imwrite(output_path, volume, dtype=np.float32)

# 获取文件列表
file_list = [f for f in os.listdir(input_dir) if f.endswith(".tif")]

# 多线程处理
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_list), total=len(file_list), desc="Processing .tif files"))

