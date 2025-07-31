import os
import re

# 文件夹路径
folder_path = 'anonymousVCD_torch_gnode05/sample/11_18_test'

# 获取文件夹中的所有 .tif 文件
files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

# 正则表达式用于匹配文件名中的数字部分
pattern = re.compile(r'(\d{8})')

# 遍历文件并重命名
for file_name in files:
    # 匹配文件名中的数字部分
    match = pattern.search(file_name)
    if match:
        new_name = match.group(1) + '.tif'
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {file_name} -> {new_name}')

print("All files have been processed and renamed.")
