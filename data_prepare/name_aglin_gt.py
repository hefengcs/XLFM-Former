import os

# 定义文件夹路径
gt_folder = 'anonymousVCD_dataset/fixed_fish/debug/Green/gt'

# 定义重命名函数
def rename_files_in_folder(folder):
    for filename in os.listdir(folder):
        # 检查文件名是否符合 Green_Recon_x.tif 格式
        if filename.startswith('Green_Recon_') and filename.endswith('.tif'):
            # 提取数字部分
            base, ext = os.path.splitext(filename)
            parts = base.split('_')
            if len(parts) == 3:
                num = int(parts[2])
                # 生成新文件名
                new_name = f"{num:08d}{ext}"
                # 获取完整路径
                old_file = os.path.join(folder, filename)
                new_file = os.path.join(folder, new_name)
                # 重命名文件
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} -> {new_file}")

# 执行重命名操作
rename_files_in_folder(gt_folder)
