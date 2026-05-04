import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import AgroClassifier

def debug_image(image_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_agro_classifier.pth")
    classes_path = os.path.join(base_dir, "core", "classes.txt")
    
    # 1. Загрузка классов
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device {device}")
    
    # 2. Загрузка модели
    model = AgroClassifier(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Подготовка фото
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 4. Предсказание
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
    
    # 5. Вывод ТОП-5 вероятностей
    print(f"\n--- DEBUG RESULTS FOR: {os.path.basename(image_path)} ---")
    top_probs, top_idxs = torch.topk(probs, k=min(5, len(classes)))
    
    for i in range(len(top_probs)):
        idx = top_idxs[i].item()
        prob = top_probs[i].item() * 100
        print(f"{i+1}. [{idx}] {classes[idx]:<40} : {prob:.2f}%")
    
    print("\n-------------------------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_image.py <path_to_image>")
    else:
        debug_image(sys.argv[1])
