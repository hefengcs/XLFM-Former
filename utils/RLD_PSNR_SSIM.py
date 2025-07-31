
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import tifffile
import numpy as np
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from src.loss import PSNRLoss, ms_ssim, edges_loss, compute_projection_loss, PyramidPoolingLoss,focal_mse_loss,_ssim,SSIMLoss
# 参数
input_dir = "/home/LifeSci/wenlab/hefengcs/VCD_dataset/NemoS/test/RLD10"
label_dir = "/home/LifeSci/wenlab/hefengcs/VCD_dataset/NemoS/test/RLD60"
num_gpus = torch.cuda.device_count()

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".tif")])
assert len(input_files) == len(label_files), "文件数不一致"

file_pairs = list(zip(input_files, label_files))

# 分配任务到每个 GP
def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# 单进程评估
def evaluate_on_gpu(rank, file_pair_list, return_dict):
    device = torch.device(f"cuda:{rank}")
    ssim_fn = SSIMLoss().to(device)
    psnr_fn = PSNRLoss().to(device)

    ssim_vals = []
    psnr_vals = []

    for input_file, label_file in tqdm(file_pair_list, desc=f"GPU {rank}"):
        input_path = os.path.join(input_dir, input_file)
        label_path = os.path.join(label_dir, label_file)

        input_np = tifffile.imread(input_path).astype(np.float32)
        label_np = tifffile.imread(label_path).astype(np.float32)

        # 只保留值大于 1 的区域，其他设为 0
        mask = (label_np > 10).astype(np.float32)
        input_np = input_np * mask
        label_np = label_np * mask


        input_np = input_np/2000
        label_np = label_np/2000

        if input_np.ndim == 3:
            slices = zip(input_np, label_np)
        else:
            slices = [(input_np, label_np)]

        for input_slice, label_slice in slices:
            input_tensor = torch.from_numpy(input_slice).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.from_numpy(label_slice).unsqueeze(0).unsqueeze(0).to(device)

            ssim_val = ssim_fn(input_tensor, label_tensor).item()
            psnr_val = -psnr_fn(input_tensor, label_tensor).item()
            ssim_vals.append(ssim_val)
            psnr_vals.append(psnr_val)

    return_dict[rank] = (ssim_vals, psnr_vals)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    manager = mp.Manager()
    return_dict = manager.dict()

    # 将文件分配到每个GPU
    task_splits = split_list(file_pairs, num_gpus)

    # 启动子进程
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=evaluate_on_gpu, args=(rank, task_splits[rank], return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 聚合结果
    all_ssim = []
    all_psnr = []
    for v in return_dict.values():
        all_ssim.extend(v[0])
        all_psnr.extend(v[1])

    print(f"\n最终平均 SSIM: {np.mean(all_ssim):.4f}")
    print(f"最终平均 PSNR: {np.mean(all_psnr):.2f} dB")