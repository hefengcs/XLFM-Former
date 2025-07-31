import os
import re

# 设置文件所在的目录路径
directory = 'anonymousVCD_torch_gnode05/sample/11_18_test/Green_Recon'  # 将此路径替换为你的文件路径

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 匹配文件名中的前导零部分，并将零去掉，保留实际数字部分
    #match = re.search(r'Green_Recon_0*(\d+)\.mat', filename)
    match = re.search(r'0*(\d+)\.mat', filename)
    if match:
        # 获取数字部分，去掉前导零
        new_number = match.group(1)  # 提取匹配到的数字部分
        new_filename = f'Green_Recon_{new_number}.mat'  # 构建新的文件名

        # 获取文件的完整路径
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')