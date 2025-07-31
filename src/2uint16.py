import tifffile as tiff
import numpy as np

image =tiff.imread('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/input/00000824.tif_epoch250.tif')
#将小于0的部分设置为0
image[image<0] = 0

obj_recon_uint16 = ((image) *65535).astype(np.uint16)


tiff.imwrite('/gpfs/home/LifeSci/wenlab/hefengcs/VCD5.12/VCD/RLD/input/new_uint1600000824.tif_epoch250.tif',obj_recon_uint16)