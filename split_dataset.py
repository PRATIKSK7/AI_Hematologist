import os
import shutil
import random
from pathlib import Path

# Paths
INPUT_DIR = Path('datasets/morphology_dataset')
OUTPUT_DIR = Path('datasets/morphology_split')

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Categories to ignore
IGNORE_CLASSES = {'Uncategorised', 'Teardrop'}

def split_dataset():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    # Create base output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # Get all subdirectories (classes) in the input folder
    classes = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

    total_copied = 0

    for cls_dir in classes:
        cls_name = cls_dir.name
        
        # Skip ignored classes
        if cls_name in IGNORE_CLASSES:
            print(f"Skipping ignored class: {cls_name}")
            continue

        # Get all files in the class directory
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        
        if not files:
            print(f"Warning: No files found in class {cls_name}. Skipping.")
            continue

        # Shuffle files to ensure randomness
        random.seed(42) # Set seed for reproducibility
        random.shuffle(files)

        total_files = len(files)
        
        # Calculate split indices
        train_end = int(total_files * TRAIN_RATIO)
        val_end = train_end + int(total_files * VAL_RATIO)

        # Split files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Copy files to respective directories
        file_splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        print(f"Processing class: {cls_name} ({total_files} files) -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        for split, split_files in file_splits.items():
            split_cls_dir = OUTPUT_DIR / split / cls_name
            split_cls_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in split_files:
                dest_path = split_cls_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                total_copied += 1

    print(f"\nDataset split complete. Copied a total of {total_copied} files to {OUTPUT_DIR}.")

if __name__ == "__main__":
    split_dataset()
