import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path("a:/AgroScan-AI")
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Mapping for Tomato classes from archive (4)
tomato_mapping = {
    "Tomato_Early_blight_leaf": "Tomato_Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_leaf": "Tomato_healthy",
    "Tomato_leaf_bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato_Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato__Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_mold_leaf": "Tomato_Leaf_Mold",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato_Spider_mites_Two_spotted_spider_mite"
}

ARCHIVE_4_DIR = BASE_DIR / "model/archive (4)"
RICE_DIR = BASE_DIR / "model/archive (1)/rice_leaf_diseases"

def copy_images(src_dir, dst_dir, prefix=""):
    if not src_dir.exists():
        print(f"Directory not found: {src_dir}")
        return
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.PNG")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.JPEG"))
    count = 0
    for file in files:
        new_name = prefix + file.name
        dst_file = dst_dir / new_name
        
        # Ensure unique filename
        suffix_count = 1
        while dst_file.exists():
            dst_file = dst_dir / f"{dst_file.stem}_{suffix_count}{dst_file.suffix}"
            suffix_count += 1
            
        shutil.copy2(file, dst_file)
        count += 1
    print(f"Copied {count} images from {src_dir.name} to {dst_dir.name}")

# 1. Copy Archive (4) Tomato classes
print("Processing Archive (4) Tomato classes...")
for src_name, target_name in tomato_mapping.items():
    # Train
    src_train = ARCHIVE_4_DIR / "train" / src_name
    dst_train = TRAIN_DIR / target_name
    if src_train.exists():
        copy_images(src_train, dst_train, prefix="a4_")
        
    # Val (from test)
    src_test = ARCHIVE_4_DIR / "test" / src_name
    dst_val = VAL_DIR / target_name
    if src_test.exists():
        copy_images(src_test, dst_val, prefix="a4_")

# 2. Copy and split Rice classes
print("\nProcessing Rice classes...")
rice_classes = {
    "Bacterial leaf blight": "Rice_Bacterial_leaf_blight",
    "Brown spot": "Rice_Brown_spot",
    "Leaf smut": "Rice_Leaf_smut"
}

for src_name, target_name in rice_classes.items():
    src_class_dir = RICE_DIR / src_name
    if not src_class_dir.exists():
        print(f"Directory not found: {src_class_dir}")
        continue
        
    files = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.JPG")) + list(src_class_dir.glob("*.png")) + list(src_class_dir.glob("*.PNG")) + list(src_class_dir.glob("*.jpeg")) + list(src_class_dir.glob("*.JPEG"))
    random.shuffle(files)
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    dst_train_dir = TRAIN_DIR / target_name
    dst_val_dir = VAL_DIR / target_name
    dst_train_dir.mkdir(parents=True, exist_ok=True)
    dst_val_dir.mkdir(parents=True, exist_ok=True)
    
    for f in train_files:
        shutil.copy2(f, dst_train_dir / f"rice_{f.name}")
    for f in val_files:
        shutil.copy2(f, dst_val_dir / f"rice_{f.name}")
        
    print(f"Copied {len(train_files)} to train and {len(val_files)} to val for {target_name}")

print("\nData preparation complete!")
