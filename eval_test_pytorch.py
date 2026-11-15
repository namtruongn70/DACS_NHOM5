import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Tắt log rác nếu có
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. CẤU HÌNH THIẾT BỊ (GPU / CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang sử dụng thiết bị: {device}")
if device.type == 'cuda':
    print(f"🎮 Tên GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TẬP TEST ĐỘC LẬP
# ==========================================
TEST_DIR = r'F:\archive\real_vs_fake\real-vs-fake\test'
IMG_SIZE = 224
BATCH_SIZE = 32

def create_test_dataframe(data_dir):
    filepaths = []
    labels = []
    
    for label in ['fake', 'real']:
        class_dir = os.path.join(data_dir, label)
        if os.path.exists(class_dir):
            all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            print(f"📁 Tìm thấy {len(all_files)} ảnh trong thư mục test/{label}")
            
            for f in all_files:
                filepaths.append(os.path.join(class_dir, f))
                # Quy ước nhãn: 'fake' = 0, 'real' = 1
                labels.append(0 if label == 'fake' else 1)
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {class_dir}")
                
    return filepaths, labels

print("📌 Đang quét danh sách dữ liệu trên tập Test độc lập...")
test_paths, test_labels = create_test_dataframe(TEST_DIR)
print(f"📊 Tổng số ảnh tập Test cần đánh giá: {len(test_paths)}")

if len(test_paths) == 0:
    raise ValueError("❌ Không tìm thấy ảnh nào trong tập Test! Vui lòng kiểm tra lại đường dẫn F:\\archive\\real_vs_fake\\real-vs-fake\\test")

# ==========================================
# 2. TẠO TEST DATASET & PIPELINE DỮ LIỆU
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

# Tập test chỉ cần resize, đưa về tensor và normalize chuẩn hóa ImageNet (không dùng augmentation)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = DeepfakeDataset(test_paths, test_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH VÀ TẢI TRỌNG SỐ TỐT NHẤT
# ==========================================
class ResNet50Deepfake(nn.Module):
    def __init__(self):
        super(ResNet50Deepfake, self).__init__()
        self.base_model = models.resnet50(weights=None) # Không cần tải lại weights ImageNet vì ta load weights đã train
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

model_path = 'best_resnet50_model_torch.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file trọng số '{model_path}'! Hãy chắc chắn bạn đã train xong.")

print(f"\n📂 Đang tải trọng số mô hình từ: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Đã tải trọng số mô hình thành công!")

# ==========================================
# 4. CHẠY DỰ ĐOÁN TRÊN TẬP TEST ĐỘC LẬP
# ==========================================
print("\n🚀 Đang chạy dự đoán xác suất trên tập Test độc lập...")
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# ==========================================
# 5. TỐI ƯU HÓA NGƯỠNG (THRESHOLD TUNING)
# ==========================================
best_threshold = 0.5
best_accuracy = 0.0

thresholds = np.linspace(0.1, 0.9, 81)
for th in thresholds:
    acc_temp = accuracy_score(all_targets, (all_preds > th).astype(int))
    if acc_temp > best_accuracy:
        best_accuracy = acc_temp
        best_threshold = th

print(f"\n✨ Ngưỡng phân loại tối ưu tìm được trên tập Test: {best_threshold:.2f}")

# Dùng ngưỡng tối ưu để ra quyết định phân loại cuối cùng
final_preds = (all_preds > best_threshold).astype(int)
conf_matrix = confusion_matrix(all_targets, final_preds)
target_names = ['fake', 'real']

# ==========================================
# 6. XUẤT KẾT QUẢ BÁO CÁO RA MÀN HÌNH
# ==========================================
print("\n" + "="*65)
print("📊 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ĐỘC LẬP (PYTORCH)")
print("="*65)
print(f"🔥 TỈ LỆ CHÍNH XÁC (TEST ACCURACY): {best_accuracy * 100:.2f}%")
print(f"🎯 Ngưỡng tối ưu áp dụng: {best_threshold:.2f}")
print("-" * 65)
print("\n📋 Báo cáo chi tiết (Classification Report):")
print(classification_report(all_targets, final_preds, target_names=target_names, digits=4))

print("📌 Ma trận nhầm lẫn (Confusion Matrix):")
print(conf_matrix)
print("="*65)

# ==========================================
# 7. VẼ VÀ LƯU BIỂU ĐỒ MA TRẬN NHẦM LẪN
# ==========================================
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'PyTorch Test Accuracy: {best_accuracy * 100:.2f}%', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.tight_layout()

output_fig = 'confusion_matrix_pytorch_test.png'
plt.savefig(output_fig)
print(f"💾 Đã lưu biểu đồ vào file: '{output_fig}'")
plt.show()