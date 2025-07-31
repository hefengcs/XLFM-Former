import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from .dataset  import CustomTiffDataset, compute_min_max, MinMaxNormalize
import pytorch_lightning as pl
class VCDDataModule(pl.LightningDataModule):
    def __init__(self, input_dir, gt_dir, batch_size):
        super().__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        initial_transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomTiffDataset(input_dir=self.input_dir, gt_dir=self.gt_dir, input_transform=initial_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.input_min_val, self.input_max_val, self.gt_min_val, self.gt_max_val = compute_min_max(dataloader)

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxNormalize(self.input_min_val, self.input_max_val)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxNormalize(self.gt_min_val, self.gt_max_val)
        ])

        dataset = CustomTiffDataset(input_dir=self.input_dir, gt_dir=self.gt_dir, input_transform=self.input_transform,
                                    gt_transform=self.gt_transform)
        total_size = len(dataset)
        test_size = int(0.1 * total_size)
        train_size = total_size - test_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
