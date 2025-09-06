import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SketchPhotoDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, sketch_transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sketch_transform = sketch_transform
        
        photo_dir = self.data_dir / "photo"
        sketch_dir = self.data_dir / "sketch"
        
        self.classes = sorted([d.name for d in photo_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.sketch_files = []
        self.photo_files_by_class = {}
        
        for cls in self.classes:
            sketches = list((sketch_dir / cls).glob("*"))
            photos = list((photo_dir / cls).glob("*"))
            
            if photos and sketches:
                self.sketch_files.extend([(s, cls) for s in sketches])
                self.photo_files_by_class[cls] = photos
        
        train_size = int(0.8 * len(self.sketch_files))
        val_size = len(self.sketch_files) - train_size
        
        if split == "train":
            self.sketch_files = self.sketch_files[:train_size]
        else:
            self.sketch_files = self.sketch_files[train_size:]
    
    def __len__(self):
        return len(self.sketch_files)
    
    def __getitem__(self, idx):
        sketch_path, class_name = self.sketch_files[idx]
        class_idx = self.class_to_idx[class_name]
        
        sketch = Image.open(sketch_path).convert('L')
        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        
        positive_photo = random.choice(self.photo_files_by_class[class_name])
        positive_photo = Image.open(positive_photo).convert('RGB')
        if self.transform:
            positive_photo = self.transform(positive_photo)
        
        negative_class = random.choice([c for c in self.classes if c != class_name])
        negative_photo = random.choice(self.photo_files_by_class[negative_class])
        negative_photo = Image.open(negative_photo).convert('RGB')
        if self.transform:
            negative_photo = self.transform(negative_photo)
        
        return {
            'sketch': sketch,
            'positive_photo': positive_photo,
            'negative_photo': negative_photo,
            'class_idx': class_idx
        }

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_files = list(self.image_dir.rglob("*"))
        self.image_files = [f for f in self.image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(image_path)