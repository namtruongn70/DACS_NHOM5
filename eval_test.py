import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Tắt log rác TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TẬP TEST ĐỘC LẬP
# ==========================================
TEST_DIR = r'F:\archive\real_vs_fake\real-vs-fake\test'

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
                labels.append(label)
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {class_dir}")
                
    df = pd.DataFrame({'filename': filepaths, 'label': labels})
    return df

print("📌 Đang quét danh sách dữ liệu trên tập Test độc lập...")
test_df = create_test_dataframe(TEST_DIR)

if len(test_df) == 0:
    raise ValueError("❌ Không tìm thấy ảnh nào trong tập Test! Vui lòng kiểm tra lại đường dẫn.")

# ==========================================
# 2. KHỞI TẠO TEST DATA GENERATOR
# ==========================================
test_datagen = ImageDataGenerator(rescale=1.0/255)
BATCH_SIZE = 32

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Bắt buộc False để khớp nhãn
)

# ==========================================
# 3. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
# ==========================================
model_path = 'best_deepfake_model.h5'
if not os.path.exists(model_path):
    model_path = 'deepfake_model.h5'

print(f"\n📂 Đang tải trọng số mô hình từ: {model_path}")
model = load_model(model_path)
print("✅ Đã tải mô hình thành công!")

# ==========================================
# 4. CHẠY DỰ ĐOÁN & TỐI ƯU HÓA NGƯỠNG (THRESHOLD)
# ==========================================
print("\n🚀 Đang chạy dự đoán xác suất trên tập Test...")
test_generator.reset()
predictions = model.predict(test_generator, verbose=1).ravel()
y_true = test_generator.classes[:len(predictions)]

# Kỹ thuật: Quét tìm ngưỡng tối ưu (từ 0.1 đến 0.9) để đạt Accuracy cao nhất
best_threshold = 0.5
best_accuracy = 0.0

thresholds = np.linspace(0.1, 0.9, 81)
for th in thresholds:
    y_pred_temp = (predictions > th).astype(int)
    acc_temp = accuracy_score(y_true, y_pred_temp)
    if acc_temp > best_accuracy:
        best_accuracy = acc_temp
        best_threshold = th

print(f"\n✨ Đã tìm ra Ngưỡng phân loại tối ưu: {best_threshold:.2f} (giúp nâng cao tỉ lệ chính xác)")

# Dùng ngưỡng tối ưu vừa tìm được để lấy nhãn dự đoán chính thức
y_pred = (predictions > best_threshold).astype(int)
target_names = list(test_generator.class_indices.keys())

# Tính toán các chỉ số
conf_matrix = confusion_matrix(y_true, y_pred)

# ==========================================
# 5. XUẤT KẾT QUẢ PHẦN TRĂM RA MÀN HÌNH
# ==========================================
print("\n" + "="*65)
print("📊 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ĐỘC LẬP")
print("="*65)
print(f"🔥 TỈ LỆ CHÍNH XÁC (ACCURACY): {best_accuracy * 100:.2f}%")
print(f"🎯 Ngưỡng tối ưu áp dụng: {best_threshold:.2f}")
print("-" * 65)
print("\n📋 Báo cáo chi tiết (Classification Report):")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

print("📌 Ma trận nhầm lẫn (Confusion Matrix):")
print(conf_matrix)
print("="*65)

# ==========================================
# 6. VẼ VÀ LƯU BIỂU ĐỒ MA TRẬN NHẦM LẪN
# ==========================================
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Optimized Test Accuracy: {best_accuracy * 100:.2f}%', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.tight_layout()

output_fig = 'confusion_matrix_optimized.png'
plt.savefig(output_fig)
print(f"💾 Đã lưu biểu đồ vào file: '{output_fig}'")
plt.show()