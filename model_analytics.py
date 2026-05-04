import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
from tqdm import tqdm

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import AgroInferencePipeline

def generate_analytics(test_dir="dataset/val"):
    print("🚀 Запуск аналитики обученной модели...")
    
    # Создаем папку для отчетов
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    # Инициализация пайплайна
    pipeline = AgroInferencePipeline()
    model = pipeline.classifier
    device = pipeline.device
    model.eval()
    
    classes = pipeline.class_names
    y_true = []
    y_pred = []
    
    print(f"📦 Обработка тестового датасета: {test_dir}")
    
    # Собираем предсказания
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Анализ класса {class_name} ({len(images)} фото)...")
        
        for img_name in tqdm(images, leave=False):
            img_path = os.path.join(class_path, img_name)
            try:
                # Используем препроцессинг из пайплайна
                with torch.no_grad():
                    # Быстрый способ получить индекс предсказания
                    res = pipeline.run_inference(img_path)
                    pred_idx = classes.index(res['raw_name'])
                    
                    y_true.append(class_idx)
                    y_pred.append(pred_idx)
            except Exception as e:
                continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Генерация Confusion Matrix
    print("📊 Генерация матрицы ошибок...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: AgroScan AI')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Расчет точности по каждому классу
    print("📈 Расчет точности по классам...")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    class_accuracies = {}
    for cls in classes:
        if cls in report:
            class_accuracies[cls] = report[cls]['recall'] # Recall для класса это и есть точность его нахождения

    # Сортируем и выбираем топ-10 лучших и худших
    sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    # Визуализация точности
    plt.figure(figsize=(12, 8))
    top_10 = sorted_acc[:10]
    names = [x[0] for x in top_10]
    values = [x[1] for x in top_10]
    
    plt.barh(names, values, color='#2d6a4f')
    plt.title('Top 10 Most Accurate Classes')
    plt.xlabel('Accuracy (Recall)')
    plt.xlim(0, 1.0)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'top_accuracy.png'))
    plt.close()

    # 3. Сохранение текстового отчета
    with open(os.path.join(report_dir, 'full_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== AgroScan AI Model Evaluation Report ===\n")
        f.write(classification_report(y_true, y_pred, target_names=classes))
        
    print(f"\n✅ Отчет успешно сформирован!")
    print(f"📁 Графики сохранены в папку: {os.path.abspath(report_dir)}")
    print(f"  - confusion_matrix.png (Матрица ошибок)")
    print(f"  - top_accuracy.png (Топ лучших классов)")
    print(f"  - full_report.txt (Подробные метрики)")

if __name__ == "__main__":
    # Можно передать пути через аргументы, если нужно
    test_path = "dataset/val" if os.path.exists("dataset/val") else "dataset/test"
    generate_analytics(test_dir=test_path)
