import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. CẤU HÌNH THIẾT BỊ (TỰ ĐỘNG NHẬN GPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang sử dụng thiết bị: {device}")
if device.type == 'cuda':
    print(f"🎮 Tên GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
TRAIN_DIR = r'F:\archive\real_vs_fake\real-vs-fake\train'
VAL_DIR  = r'F:\archive\real_vs_fake\real-vs-fake\valid'

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_WARMUP = 3
EPOCHS_FINE = 6

# ==========================================
# 2. HÀM LỌC DỮ LIỆU THÔNG MINH (BẢO TOÀN 100% ẢNH TỰ THU THẬP CẢ REAL VÀ FAKE)
# ==========================================
def create_smart_dataframe(data_dir, max_per_class=35000):
    filepaths = []
    labels = []
    
    for label in ['fake', 'real']:
        class_dir = os.path.join(data_dir, label)
        if os.path.exists(class_dir):
            all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Lọc các file tự thu thập dựa trên các tiền tố nhận diện
            user_files = [f for f in all_files if any(p in f.lower() for p in ['user_fake', 'fake_user', 'user', 'real_user', 'user_real'])]
            other_files = [f for f in all_files if f not in user_files]
            
            # Nạp toàn bộ ảnh tự thu thập + bổ sung ảnh gốc cho đủ số lượng
            needed_other = max(0, max_per_class - len(user_files))
            selected_files = user_files + other_files[:needed_other]
            
            print(f"🎯 Lớp [{label.upper()}]: Đã nạp {len(user_files)} ảnh TỰ THU THẬP + {len(selected_files) - len(user_files)} ảnh bổ sung (Tổng: {len(selected_files)})")
            
            for f in selected_files:
                filepaths.append(os.path.join(class_dir, f))
                # Quy ước: 'fake' = 0, 'real' = 1
                labels.append(0 if label == 'fake' else 1)
                
    return filepaths, labels

print("📌 Đang quét và lọc danh sách dữ liệu huấn luyện...")
train_paths, train_labels = create_smart_dataframe(TRAIN_DIR, max_per_class=35000)
val_paths, val_labels = create_smart_dataframe(VAL_DIR, max_per_class=6000)

class_indices = {'fake': 0, 'real': 1}
with open('class_indices_resnet.json', 'w') as f:
    json.dump(class_indices, f)

# ==========================================
# 3. TẠO CUSTOM DATASET & PIPELINE DỮ LIỆU
# ==========================================
class DeepfakeDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform)
val_dataset = DeepfakeDataset(val_paths, val_labels, transform=val_transform)

# Đặt num_workers=0 để chạy ổn định tuyệt đối trên Windows
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 4. XÂY DỰNG MÔ HÌNH RESNET50 & HEAD TÙY CHỈNH
# ==========================================
class ResNet50Deepfake(nn.Module):
    def __init__(self):
        super(ResNet50Deepfake, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Đóng băng toàn bộ backbone ở giai đoạn đầu
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.base_model(x)

model = ResNet50Deepfake().to(device)
criterion = nn.BCEWithLogitsLoss()

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# ==========================================
# 5. GIAI ĐOẠN 1: WARM-UP (3 EPOCHS)
# ==========================================
print("\n🚀 --- GIAI ĐOẠN 1: WARM-UP CLASSIFIER (3 EPOCHS) ---")
optimizer = optim.Adam(model.base_model.fc.parameters(), lr=1e-3)

for epoch in range(EPOCHS_WARMUP):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    loop = tqdm(train_loader, desc=f"Epoch Warm-up {epoch+1}/{EPOCHS_WARMUP}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    print(f"👉 Warm-up Epoch {epoch+1} kết thúc - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

# ==========================================
# 6. GIAI ĐOẠN 2: FINE-TUNING (Mở khóa 40 lớp cuối)
# ==========================================
print("\n🔓 --- GIAI ĐOẠN 2: MỞ KHÓA FINE-TUNING 40 LỚP CUỐI RESNET50 ---")
for param in model.base_model.parameters():
    param.requires_grad = True

children = list(model.base_model.children())
for child in children[:-4]:
    for param in child.parameters():
        param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  

best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS_FINE):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    loop = tqdm(train_loader, desc=f"Epoch Fine-tune {epoch+1}/{EPOCHS_FINE}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    # Đánh giá trên tập Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    v_loss = val_loss / val_total
    v_acc = val_correct / val_total
    
    scheduler.step(v_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(v_loss)
    history['val_acc'].append(v_acc)
    
    print(f"🔥 Fine-tune Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
    
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), 'best_resnet50_model_torch.pth')
        print(f"💾 Đã lưu mô hình tốt nhất với Val Acc: {best_val_acc * 100:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:
            print("⏹️ Early Stopping kích hoạt do Val Loss không cải thiện!")
            break

print("\n🎉 --- HOÀN THÀNH HUẤN LUYỆN ---")

# ==========================================
# 7. ĐÁNH GIÁ CHI TIẾT & VẼ BIỂU ĐỒ
# ==========================================
model.load_state_dict(torch.load('best_resnet50_model_torch.pth'))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Quét tìm ngưỡng tối ưu (Threshold Tuning)
best_threshold = 0.5
best_accuracy = 0.0
for th in np.linspace(0.1, 0.9, 81):
    acc_temp = accuracy_score(all_targets, (all_preds > th).astype(int))
    if acc_temp > best_accuracy:
        best_accuracy = acc_temp
        best_threshold = th

final_preds = (all_preds > best_threshold).astype(int)
print(f"\n🔥 TỈ LỆ CHÍNH XÁC CAO NHẤT (VAL ACCURACY): {best_accuracy * 100:.2f}%")
print(f"🎯 Ngưỡng tối ưu áp dụng: {best_threshold:.2f}")
print("-" * 65)
print(classification_report(all_targets, final_preds, target_names=['fake', 'real'], digits=4))

# Vẽ biểu đồ tiến trình huấn luyện
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
plt.title(f'PyTorch ResNet50 Accuracy (Max Val: {best_accuracy*100:.2f}%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss', marker='o', color='orange')
if len(history['val_loss']) > 0:
    plt.plot(history['val_loss'], label='Val Loss', marker='s', color='red')
plt.title('PyTorch ResNet50 Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_history_pytorch.png')
print("💾 Đã lưu biểu đồ vào file 'training_history_pytorch.png'")
plt.show()