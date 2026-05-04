import os

def generate_class_mapping(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} not found.")
        return []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    return classes

if __name__ == "__main__":
    # Определение абсолютного пути к корню проекта
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_TRAIN = os.path.join(BASE_DIR, "dataset", "train")
    OUTPUT_FILE = os.path.join(BASE_DIR, "core", "classes.txt")
    
    # Создаем папку core, если её вдруг нет
    os.makedirs(os.path.join(BASE_DIR, "core"), exist_ok=True)
    
    classes = generate_class_mapping(DATASET_TRAIN)
    if classes:
        with open(OUTPUT_FILE, "w") as f:
            for c in classes:
                f.write(c + "\n")
        print(f"Success! Generated {OUTPUT_FILE} with {len(classes)} classes.")
    else:
        print("No classes found. Check if 'dataset/train' folder contains subdirectories.")
