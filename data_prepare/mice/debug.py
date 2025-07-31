#!/usr/bin/env python3
# clip_tif_10_100.py

import numpy as np
import tifffile as tiff
from pathlib import Path

# ----------- 路径设置 -----------
src_path  = Path("anonymousVCD_dataset/mice/mice1/sub1/g/00022821.tif")
dst_dir   = Path("anonymousVCD_dataset/mice/mice1/sub1/test")
dst_path  = dst_dir / src_path.name

# ----------- 读取图像 -----------
img = tiff.imread(src_path)          # 支持 2-D / 3-D / 多通道

# ----------- 截断到 10–100 -----------
img_clipped = np.clip(img, 10, 100).astype(img.dtype)  # 保持原数据类型

# ----------- 保存 -----------
dst_dir.mkdir(parents=True, exist_ok=True)
tiff.imwrite(dst_path, img_clipped)

print(f"[✓] Saved clipped image to: {dst_path}")
