#**Tuần 3: Đánh giá mô hình & Tinh chỉnh chuyên sâu**
##**Việc đã làm:**

Xây dựng Pipeline đánh giá tự động sau khi train: Sử dụng đoạn code suy luận trên val_generator (với shuffle=False) để lấy vector nhãn thực tế (y_true) đối chiếu với vector dự đoán (y_pred dựa trên ngưỡng xác suất 0.5), từ đó xuất ra báo cáo chi tiết Classification Report và Confusion Matrix (sklearn.metrics).

Thiết kế lọc dữ liệu thông minh (create_smart_dataframe): Quản lý nghiêm ngặt 60.000 ảnh train và 10.000 ảnh validation, trong đó gom toàn bộ 100% ảnh tự thu thập của 30 thành viên vào tập train nhằm định hướng mô hình quen thuộc với môi trường thực tế của nhóm.

Tinh chỉnh tăng cường dữ liệu chống nhiễu Video Call (ImageDataGenerator): Áp dụng các kỹ thuật augmentation nâng cao gồm dịch chuyển (width/height_shift_range=0.15), co giãn (zoom_range=0.2), lật ngang (horizontal_flip=True) và đặc biệt là dải chỉnh độ sáng rộng (brightness_range=[0.7, 1.3]) để mô hình thích nghi tốt với điều kiện ánh sáng kém hoặc biến động khi gọi video trực tuyến.

Chiến lược huấn luyện 2 giai đoạn (Warm-up & Fine-Tuning): Khóa thân MobileNetV2 để huấn luyện nhanh tầng phân loại (3 epochs với lr=1e-3), sau đó mở khóa 40 lớp cuối (base_model.trainable = True) kết hợp Learning Rate siêu nhỏ (lr=1e-5) để mô hình tinh chỉnh sâu các đặc trưng vi mô. Đồng thời sử dụng hệ thống Callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping) để tự động lưu lại file trọng số tối ưu nhất vào best_deepfake_model.h5.

#**Khó khăn gặp phải:**

Thách thức đối với công nghệ tạo giả đời mới: Dù đã áp dụng Augmentation và Fine-Tuning 40 lớp cuối, mô hình đôi khi vẫn gặp khó khăn (dễ sinh lỗi nhận diện nhầm - False Positive) đối với các công nghệ Face Swap thế hệ mới (như Roop/ReActor) có vùng viền mặt cực kỳ mờ, ít để lại lộ lỗi ghép thô hay các ảnh chân dung nghệ thuật qua chỉnh sửa quá mịn màng.

Hiện tượng Overfitting tiềm ẩn: Dù đã thêm các lớp Dropout(0.4) và Dropout(0.3) sau lớp GlobalAveragePooling2D và Dense(128), việc phân phối lại tỷ lệ dữ liệu lớn đòi hỏi thời gian huấn luyện từng epoch khá dài trên tập dữ liệu 60.000 ảnh.

#**Định hướng tuần tới:**

Bắt đầu tiến hành xây dựng toàn bộ ứng dụng web tương tác tích hợp bằng Streamlit, sử dụng file trọng số tốt nhất best_deepfake_model.h5 và file ánh xạ nhãn class_indices.json vừa được xuất ra từ quá trình huấn luyện để đưa vào thử nghiệm thực tế.
