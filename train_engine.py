import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import copy
import csv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import AgroClassifier, AgroRealTime


class AgroTrainer:
    def __init__(self, data_root, model_type='classifier', batch_size=32, device=None):
        self.data_root = data_root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 🚀 GPU setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.device.type == "cuda":
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️ CPU mode (slow)")

        self.batch_size = batch_size
        self.model_type = model_type

        self.history_file = os.path.join(self.base_dir, 'models', f'history_{model_type}.csv')

        self.prepare_data()

        num_classes = len(self.train_dataset.classes)

        self.model = (
            AgroClassifier(num_classes=num_classes)
            if model_type == 'classifier'
            else AgroRealTime(num_classes=num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        # ⚡ AMP (ускорение GPU)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def prepare_data(self):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'train'), transform_train)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'val'), transform_val)

        use_gpu = self.device.type == "cuda"

        # 🚀 FIXED DataLoader (ускорение x3–x5)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 if use_gpu else 0,
            pin_memory=use_gpu,
            persistent_workers=use_gpu
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if use_gpu else 0,
            pin_memory=use_gpu,
            persistent_workers=use_gpu
        )

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

    def train(self, num_epochs=15):
        best_acc = 0.0
        best_model = copy.deepcopy(self.model.state_dict())

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                    loader = self.train_loader
                else:
                    self.model.eval()
                    loader = self.val_loader

                total_loss = 0
                correct = 0

                # Обертываем загрузчик в tqdm для отображения прогресса
                pbar = tqdm(loader, desc=f"{phase} Epoch {epoch+1}", unit="batch", leave=False)
                
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == "train"):
                        # ⚡ AMP ускорение
                        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            preds = outputs.argmax(dim=1)

                        if phase == "train":
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                    total_loss += loss.item() * inputs.size(0)
                    correct += (preds == labels).sum().item()
                    
                    # Обновляем информацию в прогресс-баре
                    pbar.set_postfix({'loss': loss.item(), 'acc': (preds == labels).float().mean().item()})

                epoch_loss = total_loss / len(loader.dataset)
                epoch_acc = correct / len(loader.dataset)

                print(f"{phase}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")

                if phase == "val":
                    self.scheduler.step(epoch_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(self.model.state_dict())

                        save_dir = os.path.join(self.base_dir, "models")
                        os.makedirs(save_dir, exist_ok=True)

                        torch.save(best_model,
                                    os.path.join(save_dir, f"best_agro_{self.model_type}.pth"))

        print(f"\n🔥 Best Accuracy: {best_acc:.4f}")
        self.model.load_state_dict(best_model)
        return self.model


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset")

    trainer = AgroTrainer(
        data_root=DATASET_PATH,
        model_type="classifier",
        batch_size=32  # GPU: 32 / CPU: 16
    )

    trainer.train(num_epochs=20)