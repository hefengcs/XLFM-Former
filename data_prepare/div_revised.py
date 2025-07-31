def add_prefix_to_file(input_file, output_file, prefix):
    # 打开输入文件并读取每一行
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # 打开输出文件并写入修改后的内容
    with open(output_file, 'w') as outfile:
        for line in lines:
            # 去除行末的换行符并在前面加上指定的路径
            new_line = prefix + line.strip() + '\n'
            outfile.write(new_line)

# 设置输入文件路径和输出文件路径
input_file = 'anonymousVCD_torch_gnode05/data/data_div/fish2/val_total.txt'
output_file = input_file

# 需要添加的前缀
prefix = 'anonymousVCD_dataset/moving_fish2/input_location/'

# 调用函数
add_prefix_to_file(input_file, output_file, prefix)

print(f"处理完成，输出文件保存在 {output_file}")
