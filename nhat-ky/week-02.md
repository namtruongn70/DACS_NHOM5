Nội dung trọng tâm: Xây dựng pipeline dữ liệu hybrid, huấn luyện MobileNetV2 (2 giai đoạn) và phát triển Web Streamlit nhận diện ảnh.

1. Dữ liệu & Tiền xử lý

Tập Train (60.000 ảnh): 30.000 Real + 30.000 Fake (bao gồm 1.049 ảnh tự thu thập user_fake và ảnh từ Kaggle).

Tập Validation (10.000 ảnh): 5.000 Real + 5.000 Fake độc lập.

Augmentation: Tăng cường chống nhiễu (xoay, dịch, lật, điều chỉnh độ sáng [0.7, 1.3]) phục vụ môi trường thực tế/Video Call.
<img width="598" height="290" alt="{1A0DE308-46DB-4C9E-8F98-F3B79090C8E1}" src="https://github.com/user-attachments/assets/14301ba7-00de-4e1b-8876-846b17bbdc0f" />


2. Trích xuất đặc trưng (Feature Extraction) & Kiến trúc Mô hình

Sử dụng MobileNetV2 (include_top=False) kết hợp GlobalAveragePooling2D để bóc tách và chuyển đổi ma trận ảnh thành vector đặc trưng 1 chiều. Các đặc trưng trọng tâm được mô hình lựa chọn và học hỏi bao gồm:

Đặc trưng hình học & cấu trúc khuôn mặt: Tỷ lệ khoảng cách, vị trí tương đối giữa mắt, mũi, miệng, viền hàm và cấu trúc xương mặt.

Đặc trưng kết cấu vi mô (Micro-textures): Độ phân bố chi tiết vùng da, độ sắc nét của tóc, lông mi, và các phản xạ ánh sáng (specular reflections) trong tròng mắt.

Dấu vết tạo giả điểm ảnh (Artifacts & Blending Errors): Các vết gợn kỹ thuật số, độ mịn da bất thường do GANs (StyleGAN) tạo ra, hoặc lỗi lộ viền, sai lệch màu sắc tại vùng tiếp giáp giữa mặt và cổ/phông nền do thuật toán hoán đổi khuôn mặt (Face Swap) để lại.

3. Quy trình Huấn luyện 2 Giai đoạn

Giai đoạn 1 (Warm-up - 3 Epochs): Đóng băng base model, chỉ huấn luyện lớp phân loại (lr = 1e-3). Đạt độ chính xác validation ~78.94%.

Giai đoạn 2 (Fine-Tuning - 10 Epochs): Mở khóa 40 lớp cuối của MobileNetV2 với lr = 1e-5 và các callback (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping).

Kết quả tối ưu (Epoch 7): val_loss: 0.2991 | val_accuracy: 87.56% | val_auc: 0.9704.

Báo cáo kiểm thử (10k ảnh): Độ chính xác tổng thể (Accuracy) đạt 88%; Độ chính xác nhận diện lớp FAKE (Precision) đạt 97%.
<img width="844" height="640" alt="{65340D4B-9480-4EF7-8F62-445F6EF47504}" src="https://github.com/user-attachments/assets/e45dbab3-01af-4b42-b178-6b0717e8b7a6" />


4. Phát triển Ứng dụng Web

Xây dựng giao diện Streamlit tích hợp bộ phát hiện khuôn mặt YuNet để cắt sát vùng mặt (224x224), truyền vào file trọng số tốt nhất (best_deepfake_model.h5) để trả kết quả Real/Fake trực quan.
<img width="1915" height="1080" alt="{425FC9DF-8CED-430F-8653-8F848A7FF933}" src="https://github.com/user-attachments/assets/761c508f-4542-49e4-b55b-d9661694fc1d" />
