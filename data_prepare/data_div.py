import os
import random

# 输入文件夹路径列表
# folders = [
#     "anonymousVCD_dataset/moving_fish2/input_location_clean",
#     "anonymousVCD_dataset/moving_fish3/input_location_clean",
#     "anonymousVCD_dataset/moving_fish4/input_location_clean",
#     "anonymousVCD_dataset/fixed_fish/240529/input_location_clean",
#     "anonymousVCD_dataset/fixed_fish/240724/input_location_clean",
#     "anonymousVCD_dataset/fixed_fish/240725_03/input_location_clean",
#     "anonymousVCD_dataset/fixed_fish/f6/input_location_clean"
# ]

folders = [
     "anonymousVCD_dataset/moving_fish2/input_location_case3",
     "anonymousVCD_dataset/moving_fish3/input_location_case3",
     "anonymousVCD_dataset/moving_fish4/input_location_case3",

     "anonymousVCD_dataset/fixed_fish/240529/input_location_case3",
     "anonymousVCD_dataset/fixed_fish/240724/input_location_case3",
    "anonymousVCD_dataset/fixed_fish/240725_01/input_location_case3",
     "anonymousVCD_dataset/fixed_fish/240725_03/input_location_case3",
    "anonymousVCD_dataset/fixed_fish/240903-2/input_location_case3",
     "anonymousVCD_dataset/fixed_fish/f6/input_location_case3",
     "anonymousVCD_dataset/fixed_fish/f7/input_location_case3",


]



# 存储训练集文件名的 txt 文件路径
train_txt = "anonymousVCD_torch_gnode05/data/data_div/view_mask/train_all.txt"

# 初始化训练集文件列表
train_files = []

# 遍历每个文件夹，从中选择 100 个 .tif 文件
for folder in folders:
    # 获取当前文件夹中的所有 .tif 文件
    tif_files = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".tif")]

    # 随机选择 100 个文件（如果文件数量不足 100，就选择全部）
    sampled_files = random.sample(tif_files, min(len(tif_files), 9999))

    # 将选中的文件添加到训练集列表
    train_files.extend(sampled_files)

# 将训练集文件列表写入 train.txt
with open(train_txt, 'w') as f:
    for file in train_files:
        f.write(file + '\n')

print(f"Training set created with {len(train_files)} files. Saved to {train_txt}")
