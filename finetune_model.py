import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import copy
from tqdm import tqdm

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import AgroClassifier

class AgroFinetuner:
    def __init__(self, data_root, old_model_path, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.data_root = data_root
        self.old_model_path = old_model_path
        self.batch_size = batch_size
        
        # 1. Подготовка данных
        self.prepare_data()
        
        # 2. Инициализация новой модели (19 классов)
        num_classes = len(self.train_dataset.classes)
        print(f"New total classes: {num_classes}")
        self.model = AgroClassifier(num_classes=num_classes, pretrained=True).to(self.device)
        
        # 3. Умная загрузка старых весов
        self.load_partial_weights()

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'train'), transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'val'), transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def load_partial_weights(self):
        if os.path.exists(self.old_model_path):
            print(f"Loading weights from {self.old_model_path}")
            old_state = torch.load(self.old_model_path, map_location=self.device)
            new_state = self.model.state_dict()
            
            # Переносим только те веса, которые совпадают по форме (весь бэкбон)
            for name, param in old_state.items():
                if name in new_state and param.shape == new_state[name].shape:
                    new_state[name].copy_(param)
            
            self.model.load_state_dict(new_state)
            print("Successfully loaded backbone weights. Head is initialized for 19 classes.")
        else:
            print("Old model not found, starting from pretrained EfficientNet.")

    def train(self, epochs=10):
        criterion = nn.CrossEntropyLoss()
        # Низкий learning rate, чтобы не "сломать" старые знания
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-2)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            for phase in ['train', 'val']:
                if phase == 'train': self.model.train()
                else: self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                pbar = tqdm(self.train_loader if phase == 'train' else self.val_loader, desc=phase)
                
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix({'loss': loss.item()})
                
                epoch_acc = running_corrects.double() / len(self.train_dataset if phase == 'train' else self.val_dataset)
                print(f"{phase} Acc: {epoch_acc:.4f}")
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    os.makedirs(save_path, exist_ok=True)
                    final_path = os.path.join(save_path, "best_agro_classifier.pth")
                    torch.save(self.model.state_dict(), final_path)
                    print(f"⭐️ Model saved to {final_path}!")

if __name__ == "__main__":
    BASE_DIR = "a:/AgroScan-AI"
    finetuner = AgroFinetuner(
        data_root=os.path.join(BASE_DIR, "dataset"),
        old_model_path=os.path.join(BASE_DIR, "models/best_agro_classifier.pth")
    )
    finetuner.train(epochs=10)
