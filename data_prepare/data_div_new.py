import os
import random

# 输入文件夹路径列表
folders = [
     # "anonymousVCD_dataset/moving_fish2/input_location_case3",
     # "anonymousVCD_dataset/moving_fish3/input_location_case3",
     # "anonymousVCD_dataset/moving_fish4/input_location_case3",

    #  "anonymousVCD_dataset/fixed_fish/240529/input_location_case3",
    #  "anonymousVCD_dataset/fixed_fish/240724/input_location_case3",
    # "anonymousVCD_dataset/fixed_fish/240725_01/input_location_case3",
    #  "anonymousVCD_dataset/fixed_fish/240725_03/input_location_case3",
    # "anonymousVCD_dataset/fixed_fish/240903-2/input_location_case3",
    #  "anonymousVCD_dataset/fixed_fish/f6/input_location_case3",
    #  "anonymousVCD_dataset/fixed_fish/f7/input_location_case3",
    "anonymousVCD_dataset/NemoS/10v/subset_2/input_location_case3",
    "anonymousVCD_dataset/NemoS/15v/subset_2/input_location_case3",
    # "anonymousVCD_dataset/NemoS/5v/subset/input_location_case3",
    "anonymousVCD_dataset/NemoS/10v/subset/input_location_case3",
    "anonymousVCD_dataset/NemoS/15v/subset/input_location_case3",
    #"anonymousVCD_dataset/NemoS/20v/subset/input_location_case3",


]

# 存储训练集和测试集文件名的 txt 文件路径
train_txt = "anonymousVCD_torch_gnode05/data/data_div/total_moving2fixed/train_mid_2_NemoS.txt"
# train_txt = "anonymousVCD_torch_gnode05/data/data_div/total_moving2fixed/None.txt"
test_txt = "anonymousVCD_torch_gnode05/data/data_div/total_moving2fixed/val_mid_2_NemoS.txt"

# 初始化文件列表
all_files = []

# 遍历每个文件夹，收集所有的 .tif 文件
for folder in folders:
    # 获取当前文件夹中的所有 .tif 文件
    tif_files = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".tif")]
    all_files.extend(tif_files)

# 随机打乱所有文件
random.shuffle(all_files)

# 按 2:8 划分测试集和训练集
split_index = int(len(all_files) * 0.2)
# split_index = int(len(all_files) * 1)
test_files = all_files[:split_index]
train_files = all_files[split_index:]

# 将训练集文件列表写入 train.txt
with open(train_txt, 'w') as f:
    for file in train_files:
        f.write(file + '\n')

# 将测试集文件列表写入 test.txt
with open(test_txt, 'w') as f:
    for file in test_files:
        f.write(file + '\n')

print(f"Training set created with {len(train_files)} files. Saved to {train_txt}")
print(f"Test set created with {len(test_files)} files. Saved to {test_txt}")
