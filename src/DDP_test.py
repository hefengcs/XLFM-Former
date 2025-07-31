import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    print(f"Running basic DDP example on rank {rank}.")
    tensor = torch.ones(1).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} has data {tensor[0]}")
    cleanup()

def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens22f1'
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
