import os
#补到8位数字
# 设置目录路径
input_dir = '/home/anonymous/Richardson-Lucy-Net/VCD_torch/data/input2_300'

# 遍历目录中的所有文件
for filename in os.listdir(input_dir):
    # 检查文件是否是 .tif 文件
    if filename.endswith('.tif'):
        # 获取文件的前缀和后缀
        prefix, ext = os.path.splitext(filename)

        # 检查前缀是否是数字
        if prefix.isdigit():
            # 将前缀转换成八位数，不足的前面补0
            new_prefix = prefix.zfill(8)

            # 构造新的文件名
            new_filename = new_prefix + ext

            # 获取完整的旧文件路径和新文件路径
            old_file = os.path.join(input_dir, filename)
            new_file = os.path.join(input_dir, new_filename)

            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')

print('文件重命名完成。')
