import os

# 指定目标路径
target_path = "/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/240903-2/input_location_clean"

# 遍历文件夹
for root, dirs, files in os.walk(target_path):
    for file_name in files:
        # 检查是否是 .tif 文件
        if file_name.endswith(".tif"):
            # 查找第一个出现的 'f5'
            new_name = file_name.replace("f5", "f5_", 1)  # 删除第一个 'f5'
            if new_name != file_name:
                # 获取完整路径
                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, new_name)

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
