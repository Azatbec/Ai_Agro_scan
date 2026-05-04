import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
from tqdm import tqdm

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import AgroClassifier

def evaluate_model(dataset_type="val"):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", dataset_type)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_agro_classifier.pth")
    CLASSES_PATH = os.path.join(BASE_DIR, "core", "classes.txt")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Загрузка имен классов для определения количества
    if not os.path.exists(CLASSES_PATH):
        print(f"Error: Classes file not found at {CLASSES_PATH}. Please run gen_mapping.py")
        return
        
    with open(CLASSES_PATH, 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]
    
    num_classes = len(all_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating model with {num_classes} classes on {dataset_type} set...")

    # 1. Подготовка данных
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Чтобы ImageFolder сопоставил папки правильно, мы должны убедиться, 
    # что классы в ImageFolder совпадают с нашими индексами.
    # ImageFolder.find_classes находит папки и сортирует их.
    test_dataset = datasets.ImageFolder(DATASET_PATH, transform)
    
    # Проверка на совпадение классов (опционально, но полезно для отладки)
    if test_dataset.classes != all_classes:
        print("Warning: Dataset classes don't match core/classes.txt exactly.")
        print(f"Dataset has {len(test_dataset.classes)} folders, model expects {num_classes}.")
        # Если папок меньше, ImageFolder будет использовать другие индексы.
        # Нужно переопределить class_to_idx
        class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes)}
        test_dataset.class_to_idx = class_to_idx
        # Но ImageFolder.samples строится при инициализации. Придется пересобрать.
        test_dataset.samples = test_dataset.make_dataset(test_dataset.root, class_to_idx, test_dataset.extensions, None)
        test_dataset.targets = [s[1] for s in test_dataset.samples]

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 2. Загрузка модели
    model = AgroClassifier(num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    model.eval()

    # 3. Тестирование
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Results ({dataset_type} set):")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    dtype = sys.argv[1] if len(sys.argv) > 1 else "val"
    evaluate_model(dtype)
