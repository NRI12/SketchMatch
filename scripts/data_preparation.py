import os
import gdown
import zipfile
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

def download_data():
    data_dir = Path("data")
    if data_dir.exists() and (data_dir / "sketch_dataset").exists():
        print("Dataset already exists")
        return
    
    print("Downloading dataset...")
    url = "https://drive.google.com/uc?id=1-XjOrxu9XYQyZWXNOqLReJFKc8pkJ0__"
    output = "sketch_dataset.zip"
    
    gdown.download(url, output, quiet=False)
    
    print("Extracting...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("data")
    
    os.remove(output)
    print("Dataset ready!")

def analyze_dataset():
    dataset_path = Path("data/sketch_dataset")
    
    if not dataset_path.exists():
        print("Dataset not found!")
        return
    
    photo_dir = dataset_path / "photo"
    sketch_dir = dataset_path / "sketch"
    
    classes = sorted([d.name for d in photo_dir.iterdir() if d.is_dir()])
    print(f"Classes: {len(classes)}")
    print(f"Class names: {classes[:10]}...")
    
    stats = defaultdict(dict)
    total_photos = total_sketches = 0
    
    for cls in classes:
        photo_count = len(list((photo_dir / cls).glob("*")))
        sketch_count = len(list((sketch_dir / cls).glob("*")))
        
        stats[cls] = {"photos": photo_count, "sketches": sketch_count}
        total_photos += photo_count
        total_sketches += sketch_count
    
    print(f"Total photos: {total_photos}")
    print(f"Total sketches: {total_sketches}")
    print(f"Avg photos/class: {total_photos//len(classes)}")
    print(f"Avg sketches/class: {total_sketches//len(classes)}")
    
    return classes, stats

def visualize_samples(classes):
    dataset_path = Path("data/sketch_dataset")
    
    for sample_class in classes:
        photo_dir = dataset_path / "photo" / sample_class
        sketch_dir = dataset_path / "sketch" / sample_class
        
        photo_files = list(photo_dir.glob("*"))
        sketch_files = list(sketch_dir.glob("*"))
        
        if photo_files and sketch_files:
            photo_file = photo_files[0]
            sketch_file = sketch_files[0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            photo = Image.open(photo_file)
            ax1.imshow(photo)
            ax1.set_title(f"Photo: {sample_class}")
            ax1.axis('off')
            
            sketch = Image.open(sketch_file)
            ax2.imshow(sketch, cmap='gray')
            ax2.set_title(f"Sketch: {sample_class}")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig("sample_data.png", dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Photo size: {photo.size}")
            print(f"Sketch size: {sketch.size}")
            return
    
    print("No valid samples found")

if __name__ == "__main__":
    # download_data()
    classes, stats = analyze_dataset()
    if classes:
        visualize_samples(classes)
    print("Data exploration complete!")