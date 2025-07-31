import torch
import pytorch_lightning as pl
import torch.optim as optim
from model.model import UNet, UNet_VCD
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.dataset import MinMaxNormalize
import os
import tifffile as tiff
class VCDModel(pl.LightningModule):
    def __init__(self, model_name, lf_extra, n_slices, output_size, input_min_val, input_max_val, gt_min_val,
                 gt_max_val, lr, sample_dir, channels_interp=128, normalize_mode='percentile'):
        super(VCDModel, self).__init__()
        if model_name == 'UNet':
            self.model = UNet(lf_extra, n_slices, output_size, channels_interp, normalize_mode)
        elif model_name == 'UNet_VCD':
            self.model = UNet_VCD(lf_extra, n_slices, output_size, channels_interp, normalize_mode)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.criterion = nn.MSELoss()
        self.lr = lr
        self.sample_dir = sample_dir
        self.validation_outputs = []
        self.training_outputs = []

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxNormalize(input_min_val, input_max_val)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxNormalize(gt_min_val, gt_max_val)
        ])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        mse_loss = self.criterion(outputs, labels)
        total_loss = mse_loss

        self.training_outputs.append(total_loss.detach())

        self.log('train/loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train/mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, img_names = batch
        outputs = self(inputs)
        mse_loss = self.criterion(outputs, labels)
        total_loss = mse_loss

        self.validation_outputs.append({
            "val_loss": total_loss.detach(),
            "outputs": outputs.detach(),
            "inputs": inputs.detach(),
            "input_filenames": img_names
        })

        self.log('val/loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return total_loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        self.log('val/loss_epoch', avg_val_loss, on_epoch=True, sync_dist=True)

        if self.current_epoch % 10 == 0:
            sample_dir = os.path.join(self.sample_dir, f'epoch_{self.current_epoch}')
            os.makedirs(sample_dir, exist_ok=True)
            for i, output in enumerate(self.validation_outputs):
                outputs = output["outputs"].cpu().numpy()
                input_filenames = output["input_filenames"]
                for j, input_filename in enumerate(input_filenames):
                    output_filename = f"sample_{os.path.basename(input_filename).split('.')[0]}_epoch_{self.current_epoch}.tif"
                    output_path = os.path.join(sample_dir, output_filename)
                    tiff.imwrite(output_path, outputs[j], compression="deflate")

        self.validation_outputs.clear()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        self.training_outputs.clear()
