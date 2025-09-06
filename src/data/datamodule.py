import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .datasets import SketchPhotoDataset
from .transforms import get_photo_transforms, get_sketch_transforms

class SketchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/sketch_dataset",
        batch_size=32,
        num_workers=4,
        image_size=224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage=None):
        photo_transform_train = get_photo_transforms(self.image_size, "train")
        photo_transform_val = get_photo_transforms(self.image_size, "val")
        sketch_transform_train = get_sketch_transforms(self.image_size, "train")
        sketch_transform_val = get_sketch_transforms(self.image_size, "val")
        
        self.train_dataset = SketchPhotoDataset(
            self.data_dir,
            split="train",
            transform=photo_transform_train,
            sketch_transform=sketch_transform_train
        )
        
        self.val_dataset = SketchPhotoDataset(
            self.data_dir,
            split="val", 
            transform=photo_transform_val,
            sketch_transform=sketch_transform_val
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )