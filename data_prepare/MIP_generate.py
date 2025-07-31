import os
import numpy as np
import tifffile as tiff

# 定义读取的路径
tif_dir = '/gpfsanonymousVCD_torch_gnode05/sample/infer'

# 获取目录下所有的 tif 文件
tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]

# 确保有tif文件
if len(tif_files) == 0:
    raise FileNotFoundError("No .tif files found in the directory.")

# 处理每一个tif文件
for tif_file in tif_files:
    # 读取每个 tif 文件
    tif_path = os.path.join(tif_dir, tif_file)
    tif_data = tiff.imread(tif_path)

    # 计算最大投影：XY, XZ, YZ方向的最大投影
    max_projection_xy = np.max(tif_data, axis=0)  # 沿z轴方向投影，生成XY平面的最大投影
    max_projection_xz = np.max(tif_data, axis=1)  # 沿y轴方向投影，生成XZ平面的最大投影
    max_projection_yz = np.max(tif_data, axis=2)  # 沿x轴方向投影，生成YZ平面的最大投影

    # 保存每个方向的最大投影为独立的TIF文件
    output_xy_path = os.path.join(tif_dir, f'{os.path.splitext(tif_file)[0]}_xy_projection.tif')
    output_xz_path = os.path.join(tif_dir, f'{os.path.splitext(tif_file)[0]}_xz_projection.tif')
    output_yz_path = os.path.join(tif_dir, f'{os.path.splitext(tif_file)[0]}_yz_projection.tif')

    tiff.imwrite(output_xy_path, max_projection_xy)
    tiff.imwrite(output_xz_path, max_projection_xz)
    tiff.imwrite(output_yz_path, max_projection_yz)

    print(f"Projections saved for {tif_file}:")
    print(f"  XY projection saved to {output_xy_path}")
    print(f"  XZ projection saved to {output_xz_path}")
    print(f"  YZ projection saved to {output_yz_path}")
