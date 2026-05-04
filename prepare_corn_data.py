import os
import shutil
import random

def prepare_new_data():
    base_dir = "a:/AgroScan-AI"
    source_dir = os.path.join(base_dir, "model/archive/data")
    target_train_dir = os.path.join(base_dir, "dataset/train")
    target_val_dir = os.path.join(base_dir, "dataset/val")
    
    classes_map = {
        "Blight": "Corn_Blight",
        "Common_Rust": "Corn_Common_Rust",
        "Gray_Leaf_Spot": "Corn_Gray_Leaf_Spot",
        "Healthy": "Corn_healthy"
    }
    
    for src_name, target_name in classes_map.items():
        src_path = os.path.join(source_dir, src_name)
        if not os.path.exists(src_path):
            print(f"Skipping {src_name}, not found.")
            continue
            
        # Создаем папки
        os.makedirs(os.path.join(target_train_dir, target_name), exist_ok=True)
        os.makedirs(os.path.join(target_val_dir, target_name), exist_ok=True)
        
        # Список файлов
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        print(f"Moving {len(train_files)} to train and {len(val_files)} to val for {target_name}")
        
        for f in train_files:
            shutil.copy2(os.path.join(src_path, f), os.path.join(target_train_dir, target_name, f))
        for f in val_files:
            shutil.copy2(os.path.join(src_path, f), os.path.join(target_val_dir, target_name, f))

if __name__ == "__main__":
    prepare_new_data()
