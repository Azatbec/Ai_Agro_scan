import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

def clean_class_name(name):
    # Известные соответствия (маппинги) из предыдущих датасетов
    mapping = {
        "Blight": "Corn_Blight",
        "Common_Rust": "Corn_Common_Rust",
        "Gray_Leaf_Spot": "Corn_Gray_Leaf_Spot",
        "Healthy": "Corn_healthy",
        "Tomato_Early_blight_leaf": "Tomato_Early_blight",
        "Tomato_leaf": "Tomato_healthy",
        "Tomato_leaf_bacterial_spot": "Tomato_Bacterial_spot",
        "Tomato_leaf_late_blight": "Tomato_Late_blight",
        "Tomato_leaf_mosaic_virus": "Tomato__Tomato_mosaic_virus",
        "Tomato_leaf_yellow_virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato_mold_leaf": "Tomato_Leaf_Mold",
        "Tomato_two_spotted_spider_mites_leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Bacterial leaf blight": "Rice_Bacterial_leaf_blight",
        "Brown spot": "Rice_Brown_spot",
        "Leaf smut": "Rice_Leaf_smut"
    }
    if name in mapping:
        return mapping[name]
    
    # Универсальная очистка для всех остальных классов
    clean = name.replace("___", "_").replace(" ", "_").replace("-", "_")
    return clean

def gather_all_data():
    print("1. Очистка старых данных в dataset/train и dataset/val...")
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists():
        shutil.rmtree(VAL_DIR)
        
    class_to_files = defaultdict(list)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    print("\n2. Сканирование папки model/ на наличие изображений (идет поиск во всех архивах)...")
    for root, dirs, files in os.walk(MODEL_DIR):
        root_path = Path(root)
        
        image_files = [f for f in files if Path(f).suffix.lower() in valid_extensions]
        if not image_files:
            continue
            
        raw_class_name = root_path.name
        
        # Пропускаем папки, если изображения лежат прямо в train/val/test (чтобы не было класса с именем train)
        if raw_class_name.lower() in ['train', 'val', 'test', 'valid']:
            continue
            
        class_name = clean_class_name(raw_class_name)
        
        for img in image_files:
            class_to_files[class_name].append(root_path / img)

    print(f"\nНайдено уникальных классов: {len(class_to_files)}")
    total_images = sum(len(v) for v in class_to_files.values())
    print(f"Всего изображений для копирования: {total_images}")
    
    print("\n3. Формирование нового датасета и копирование...")
    for cls_name, file_paths in tqdm(class_to_files.items(), desc="Общий прогресс по классам"):
        dst_train = TRAIN_DIR / cls_name
        dst_val = VAL_DIR / cls_name
        dst_train.mkdir(parents=True, exist_ok=True)
        dst_val.mkdir(parents=True, exist_ok=True)
        
        # Перемешиваем файлы для случайного разбиения
        random.shuffle(file_paths)
        split_idx = int(len(file_paths) * 0.8)
        
        train_files = file_paths[:split_idx]
        val_files = file_paths[split_idx:]
        
        def copy_files(files, dst_dir):
            for i, src_f in enumerate(files):
                # Создаем уникальное имя, чтобы файлы с одинаковыми именами из разных архивов не перезаписывали друг друга
                archive_name = src_f.parent.parent.name.replace(" ", "_")
                unique_name = f"{archive_name}_{i}_{src_f.name}"
                dst_f = dst_dir / unique_name
                shutil.copy2(src_f, dst_f)
                
        copy_files(train_files, dst_train)
        copy_files(val_files, dst_val)

    print("\nУСПЕШНО! Все данные из всех папок внутри model/ собраны, перемешаны и готовы к обучению.")

if __name__ == "__main__":
    gather_all_data()
