import numpy as np
import matplotlib.pyplot as plt
import pywt
import tifffile as tiff

# 加载 TIFF 文件
#tif_file = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/sample/NAFNet_pure_PSNR_RLD100_paired900_081220240812-042658/00000824.tif_epoch250.tif'
tif_file = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt900/00000824.tif'
tif_data = tiff.imread(tif_file)

# 确保数据的形状符合预期
print(tif_data.shape)

# 取出第150层
layer_150 = tif_data[150]

# 执行 Harr 变换（Haar 小波变换）
coeffs2 = pywt.dwt2(layer_150, 'haar')
LL, (LH, HL, HH) = coeffs2

# 可视化原始层和 Harr 变换后的结果
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
titles = ['Original', 'Approximation (LL)', 'Horizontal detail (LH)', 'Vertical detail (HL)', 'Diagonal detail (HH)']

axes[0].imshow(layer_150, cmap='gray')
axes[0].set_title(titles[0])
axes[0].axis('off')

axes[1].imshow(LL, cmap='gray')
axes[1].set_title(titles[1])
axes[1].axis('off')

axes[2].imshow(LH, cmap='gray')
axes[2].set_title(titles[2])
axes[2].axis('off')

axes[3].imshow(HL, cmap='gray')
axes[3].set_title(titles[3])
axes[3].axis('off')

axes[4].imshow(HH, cmap='gray')
axes[4].set_title(titles[4])
axes[4].axis('off')

plt.show()
