import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import json

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import AgroClassifier

def predict_image(image_path, model_path, classes_path):
    # 1. Загрузка имен классов
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Инициализация модели
    model = AgroClassifier(num_classes=num_classes, pretrained=False)
    
    # 3. Загрузка весов
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    model.eval()
    
    # 4. Подготовка изображения
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 5. Инференс
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probabilities, dim=0)
    
    class_name = classes[pred.item()]
    confidence = conf.item() * 100
    
    print(f"\n--- Prediction Result ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Result: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"-------------------------\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_agro_classifier.pth")
    CLASSES_PATH = os.path.join(BASE_DIR, "core", "classes.txt")
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <path_to_image>")
        # Попробуем найти тестовое изображение, если аргумент не передан
        TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
        if os.path.exists(TEST_DIR):
            for root, dirs, files in os.walk(TEST_DIR):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_img = os.path.join(root, file)
                        print(f"Using default test image: {test_img}")
                        predict_image(test_img, MODEL_PATH, CLASSES_PATH)
                        sys.exit(0)
        print("No image provided and no test images found.")
    else:
        img_path = sys.argv[1]
        predict_image(img_path, MODEL_PATH, CLASSES_PATH)
