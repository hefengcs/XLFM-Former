import numpy as np
import tifffile
import os
import glob
from tqdm import tqdm
#用于从2048，2048到600，600，27.减去背景噪声


def process_image(image_path, lenses_info, output_path):
    # 读取图像
    original_image = tifffile.imread(image_path)
    
    # 处理每个透镜区域
    cropped_images = []
    for x, y, w, h in lenses_info:
        lens_image = original_image[y:y+h, x:x+w]
        
        # 检查图像尺寸，进行必要的填充
        if lens_image.shape[0] < 600 or lens_image.shape[1] < 600:
            new_image = np.zeros((600, 600), dtype=np.float16)  # 创建一个新的空白图像
            offset_y = (600 - lens_image.shape[0]) // 2
            offset_x = (600 - lens_image.shape[1]) // 2
            new_image[offset_y:offset_y+lens_image.shape[0], offset_x:offset_x+lens_image.shape[1]] = lens_image
            cropped_images.append(new_image)
        else:
            cropped_images.append(lens_image)
    
    # 将所有裁剪后的图像堆叠成一个新的numpy数组
    stacked_images = np.stack(cropped_images, axis=0)

    stacked_images = np.maximum(stacked_images, 0)
    # 构建输出文件路径
    base_name = os.path.basename(image_path)
    output_file_path = os.path.join(output_path, base_name)
    
    # 保存为TIFF文件
    tifffile.imwrite(output_file_path, stacked_images)

# 输入和输出路径
input_path = 'anonymousVCD_dataset/fixed_fish/240903-2/g'  # 输入路径
output_path = 'anonymousVCD_dataset/fixed_fish/240903-2/input_location'  # 输出路径
coordinates_path = '/gpfsanonymousVCD_torch_gnode05/data_prepare/coordinates.txt'  # 坐标信息文件路径

# 读取坐标和尺寸信息
lenses_info = []
with open(coordinates_path, 'r') as file:
    for line in file:
        x, y, w, h = map(int, line.split())
        lenses_info.append((x, y, w, h))

# 遍历指定路径下所有的TIFF文件
for image_path in tqdm(glob.glob(os.path.join(input_path, '*.tif'))):
    process_image(image_path, lenses_info, output_path)
