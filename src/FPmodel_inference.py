import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
import tifffile as tiff
import numpy as np
from dataset import MinMaxNormalize, CustomTiffDataset_FP_Inference
from utils.head import CombinedModel_muti_stage_ImageResizer
from loss import SSIMLoss

# Lightning Module (from the training script, modified for inference)
class UNetLightningModule(pl.LightningModule):
    def __init__(self, lf_extra, n_slices, output_size, learning_rate, input_min_val, input_max_val, gt_min_val, gt_max_val):
        super(UNetLightningModule, self).__init__()
        self.model = CombinedModel_muti_stage_ImageResizer()
        self.criterion = PSNRLoss()
        self.SSIM_loss = SSIMLoss(size_average=True)
        self.learning_rate = learning_rate
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val
        self.gt_min_val = gt_min_val
        self.gt_max_val = gt_max_val
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        return x

# PSNR Loss function (required by the model)
class PSNRLoss(torch.nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return -psnr  # Returning negative PSNR for minimization

# Inference function with half-precision (FP16) support
def inference(model, input_dir, output_dir, gt_max_val, device):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the input data
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        MinMaxNormalize(model.input_min_val, model.input_max_val)
    ])

    dataset = CustomTiffDataset_FP_Inference(input_dir=input_dir, input_transform=input_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    model.to(device)

    # Use autocast for mixed precision inference
    with torch.no_grad():
        for idx, (inputs, input_filename) in enumerate(dataloader):
            inputs = inputs.to(device)

            # Enable automatic mixed precision (AMP)
            with torch.cuda.amp.autocast():
                # Forward pass through the model
                outputs = model(inputs)

            # Save output as a TIFF file
            output_filename = os.path.join(output_dir, f"{input_filename[0]}_output.tif")
            output_image = outputs.cpu().numpy().squeeze() * gt_max_val
            #维度调整
            rotated_image = np.rot90(output_image, k=1)  # k=-1 代表顺时针旋转90度

            # 再关于水平轴（H）翻转
            flipped_image = np.flipud(rotated_image)  # flipud 进行上下翻转

            tiff.imwrite(output_filename,  flipped_image)


            print(f"Saved output to {output_filename}")

if __name__ == "__main__":
    # Parameters and paths
    input_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/fixed_fish/train/gt'  # Path to the input data
    output_dir = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/inference_output'  # Path to save the output

    # Precomputed normalization values
    input_min_val, input_max_val, gt_min_val, gt_max_val = np.load('/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data/pre_data/FP_fish1_1500.npy')

    # Load the model with the same parameters as during training
    checkpoint_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/ckpt/train_FPmodel_pure_PSNR_conv_RLD60_fish1_1500_090820240909-160246/epoch=194-val_loss=0.0000160639.ckpt'
    lf_extra = 27
    n_slices = 300
    output_size = (600, 600)
    learning_rate = 1e-4

    # Instantiate the model
    model = UNetLightningModule(lf_extra=lf_extra, n_slices=n_slices, output_size=output_size,
                                learning_rate=learning_rate, input_min_val=input_min_val, input_max_val=input_max_val,
                                gt_min_val=gt_min_val, gt_max_val=gt_max_val)

    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

    # Load state_dict while ignoring mismatches, if needed
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    except RuntimeError as e:
        print(f"Warning: Some keys in the state_dict did not match the model: {e}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference with half precision
    inference(model, input_dir, output_dir, gt_max_val, device)
