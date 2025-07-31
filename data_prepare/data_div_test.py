import os
import random

# 路径
image_dir = "anonymousVCD_dataset/mice/mice1/sub1/input_location_case3"
train_output_file = "anonymousVCD_torch_gnode05/data/data_div/mice/mice_mini_train.txt"
test_output_file = "anonymousVCD_torch_gnode05/data/data_div/mice/mice_mini_val.txt"

# 获取前300个图像文件
all_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])[:200]
#all_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
# 打乱图像列表
random.shuffle(all_images)

# 计算划分点
train_split = int(0.8*len(all_images))

# 划分训练集和测试集
train_images = all_images[:train_split]
test_images = all_images[train_split:]

# 将训练集和测试集的路径保存到文件中
with open(train_output_file, 'w') as train_file:
    for image in train_images:
        train_file.write(f"{image}\n")

with open(test_output_file, 'w') as test_file:
    for image in test_images:
        test_file.write(f"{image}\n")

print(f"训练集和测试集已保存到 {train_output_file} 和 {test_output_file}")
