import torch
from src.data.datamodule import SketchDataModule

# Quick debug script
def debug_datamodule():
    # Initialize datamodule
    dm = SketchDataModule(
        data_dir="data/sketch_dataset",
        batch_size=4,  # Small batch for debugging
        num_workers=0  # No multiprocessing for debugging
    )
    dm.setup()
    
    # Check dataset sizes
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Classes: {len(dm.train_dataset.classes)}")
    print(f"First 5 classes: {dm.train_dataset.classes[:5]}")
    
    # Check single sample
    sample = dm.train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Sketch shape: {sample['sketch'].shape}")
    print(f"Photo shape: {sample['positive_photo'].shape}")
    print(f"Class idx: {sample['class_idx']}")
    
    # Check dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch sketch shape: {batch['sketch'].shape}")
    print(f"Batch photo shape: {batch['positive_photo'].shape}")
    print(f"Batch class_idx: {batch['class_idx']}")
    
    # Check if classes match
    print(f"\nClass distribution in batch: {batch['class_idx'].tolist()}")
    
    # Verify transforms
    print(f"Sketch range: [{batch['sketch'].min():.3f}, {batch['sketch'].max():.3f}]")
    print(f"Photo range: [{batch['positive_photo'].min():.3f}, {batch['positive_photo'].max():.3f}]")

if __name__ == "__main__":
    debug_datamodule()

