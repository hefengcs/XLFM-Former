import os
import torch

def test_single_gpu(rank):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device(f'cuda:{rank}')
    tensor = torch.ones(1).to(device)
    print(f'Running on GPU {rank}, tensor value: {tensor.item()}')

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        try:
            test_single_gpu(i)
        except RuntimeError as e:
            print(f'Error on GPU {i}: {e}')
