import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json
import os

# 1. Cấu hình đường dẫn dữ liệu
TRAIN_DIR = r'C:\Users\namtr\Downloads\archive\real_vs_fake\real-vs-fake\train'
VAL_DIR   = r'C:\Users\namtr\Downloads\archive\real_vs_fake\real-vs-fake\valid'

# 2. Hàm lọc dữ liệu thông minh: Đảm bảo 100% ảnh tự quay nằm trong tập train
def create_smart_dataframe(data_dir, max_per_class=30000):
    filepaths = []
    labels = []
    
    for label in ['fake', 'real']:
        class_dir = os.path.join(data_dir, label)
        if os.path.exists(class_dir):
            all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if label == 'fake':
                # Bắt tất cả ảnh do bạn tự thu thập (chứa tiền tố user_fake, fake_user, hoặc user)
                user_files = [f for f in all_files if any(p in f.lower() for p in ['user_fake', 'fake_user', 'user'])]
                other_files = [f for f in all_files if f not in user_files]
                
                # Nạp toàn bộ ảnh tự quay + bổ sung ảnh cũ cho đủ 30.000 ảnh FAKE
                needed_other = max(0, max_per_class - len(user_files))
                selected_files = user_files + other_files[:needed_other]
                
                print(f"🎯 Lớp [FAKE]: Đã nạp {len(user_files)} ảnh TỰ QUAY + {len(selected_files) - len(user_files)} ảnh cũ (Tổng: {len(selected_files)})")
            else:
                # Lớp REAL: Lấy 30.000 ảnh đầu tiên
                selected_files = all_files[:max_per_class]
                print(f"🎯 Lớp [REAL]: Đã nạp {len(selected_files)} ảnh")
                
            for f in selected_files:
                filepaths.append(os.path.join(class_dir, f))
                labels.append(label)
                    
    df = pd.DataFrame({'filename': filepaths, 'label': labels})
    return df

# Tạo DataFrame tập TRAIN (60.000 ảnh) và VALIDATION (10.000 ảnh)
print("📌 Đang kiểm tra và quét danh sách dữ liệu...")
train_df = create_smart_dataframe(TRAIN_DIR, max_per_class=30000)
val_df   = create_smart_dataframe(VAL_DIR, max_per_class=5000)

# 3. Augmentation tối ưu chống nhiễu Video Call & Ánh sáng kém
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

BATCH_SIZE = 32

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Lưu mapping nhãn (Ví dụ: {'fake': 0, 'real': 1}) ra file JSON
class_indices = train_generator.class_indices
print(f"📌 Class Indices Mapping: {class_indices}")
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# 4. Kiến trúc Mô hình MobileNetV2 Nâng Cấp
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),              # Chống Overfitting dữ liệu webcam
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# -------------------------------------------------------------------------
# GIAI ĐOẠN 1: WARM-UP (Huấn luyện các lớp Dense mới thêm vào)
# -------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\n🚀 --- GIAI ĐOẠN 1: WARM-UP CLASSIFIER (3 EPOCHS) ---")
model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator
)

# -------------------------------------------------------------------------
# GIAI ĐOẠN 2: FINE-TUNING KỸ LƯỠNG (Mở khóa 40 lớp cuối của MobileNetV2)
# -------------------------------------------------------------------------
print("\n🔓 --- GIAI ĐOẠN 2: MỞ KHÓA FINE-TUNING 40 LỚP CUỐI ---")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Dùng Learning Rate siêu nhỏ (1e-5) để tinh chỉnh sâu các điểm khác biệt nhỏ của Deepfake
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Tự động lưu checkpoint tốt nhất dựa trên val_loss
callbacks = [
    ModelCheckpoint('best_deepfake_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
]

print("\n🔥 --- BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC ---")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks
)

# Lưu mô hình hoàn chỉnh
model.save('deepfake_model.h5')
print("\n🎉 --- ĐÃ HOÀN THÀNH HUẤN LUYỆN KỸ LƯỠNG VÀ LƯU MÔ HÌNH ---")

# 5. Đánh giá chi tiết mô hình
val_generator.reset()
predictions = model.predict(val_generator)
y_pred = (predictions > 0.5).astype(int).ravel()
y_true = val_generator.classes[:len(y_pred)]

target_names = list(class_indices.keys())
print("\n📋 BÁO CÁO CHI TIẾT (CLASSIFICATION REPORT):")
print(classification_report(y_true, y_pred, target_names=target_names))

# Biểu đồ hiển thị kết quả
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Độ chính xác (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Mức độ mất mát (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()