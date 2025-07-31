import tifffile
import numpy as np
import cv2

# 读取32位TIFF图像
tif_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/loss_test/00000295.tif_epoch230.tif'
img = tifffile.imread(tif_path)

print("Original Image Data Type:", img.dtype)
print("Original Image Shape:", img.shape)

# 将图像数据类型转换为float64
img_float64 = img.astype(np.float64)
print("Converted Image Data Type:", img_float64.dtype)

# 线性变换调节亮度和对比度
gain = 2.0  # 增加对比度
bias = 100  # 增加亮度
adjusted_img = np.clip(gain * img_float64 + bias, 0, 65535)  # 限制范围在0到65535之间，确保不超出原始图像的范围

# 使用CLAHE进行对比度增强
img_uint8 = cv2.normalize(img_float64, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img_uint8)

# 将CLAHE处理后的图像转换回float64并调整到原始范围
clahe_img_float64 = cv2.normalize(clahe_img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.float64)

# 保存线性变换调整后的图像
save_path_linear = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/loss_test/infer_adjusted_linear.tif'
tifffile.imwrite(save_path_linear, adjusted_img.astype(np.uint16))

# 保存CLAHE调整后的图像
save_path_clahe = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/loss_test/infer_adjusted_clahe.tif'
tifffile.imwrite(save_path_clahe, clahe_img_float64.astype(np.uint16))

print(f"Linear adjusted image saved to: {save_path_linear}")
print(f"CLAHE adjusted image saved to: {save_path_clahe}")
