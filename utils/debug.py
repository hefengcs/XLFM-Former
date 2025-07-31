import numpy as np
import matplotlib.pyplot as plt
import tifffile

# 读取图像数据
volume_path = "/home/LifeSci/wenlab/hefengcs/VCD_dataset/mice/mice1/sub1/RLD60/00022881.tif"
volume = tifffile.imread(volume_path).astype(np.float32)

# 排除0值
voxels = volume[volume > 0]

# 计算百分位数阈值
percentiles = [99.0, 99.5, 99.7, 99.9]
thresholds = [np.percentile(voxels, p) for p in percentiles]

# 画直方图并标出这些阈值
plt.figure(figsize=(10, 5))
hist_range = (np.percentile(voxels, 1), np.percentile(voxels, 99))
plt.hist(voxels, bins=500, range=hist_range, color='blue', alpha=0.8)

for p, t in zip(percentiles, thresholds):
    plt.axvline(t, color='red', linestyle='--', label=f'{p}th: {t:.2f}')

plt.title("Voxel Intensity Distribution with Threshold Candidates")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
