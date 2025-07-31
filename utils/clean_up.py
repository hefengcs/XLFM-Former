import os
import shutil


def clear_directory(directory):
    """
    删除目录中的所有内容，并重新创建该目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"Cleared contents of the directory: {directory}")
    else:
        print(f"Directory does not exist: {directory}")


def main():
    directories = [
        #'/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt',
        '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/logs',
        '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/sample'
    ]

    for directory in directories:
        clear_directory(directory)


if __name__ == "__main__":
    main()
