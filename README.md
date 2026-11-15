# DACS_NHOM5
Hệ thống phát hiện Deepfake chân dung tích hợp Deep Learing

## 1. Cài đặt môi trường và thư viện

Mở terminal tại thư mục `deepfake_project`, kích hoạt môi trường ảo (`venv`) và cài đặt các gói cần thiết:

```bash
# Kích hoạt môi trường ảo trên Windows
venv\Scripts\activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

## 2. Các câu lệnh chạy huấn luyện (Training Scripts)

Huấn luyện mô hình ResNet50 (PyTorch) với chiến lược 2 giai đoạn:

```bash
python train_resnet50.py
```

Huấn luyện mô hình cơ sở ban đầu:

```bash
python train.py
```

## 3. Các câu lệnh chạy đánh giá độc lập (Evaluation Scripts)

Đánh giá mô hình ResNet50 trên tập test độc lập (xuất file confusion_matrix_pytorch_test.png):

```bash
python eval_test_pytorch.py
```

Đánh giá mô hình MobileNetV2 trên tập test độc lập (xuất file confusion_matrix_optimized.png):

```bash
python eval_test.py
```

## 4. Các câu lệnh khởi chạy ứng dụng giao diện (Streamlit Apps)

Chạy ứng dụng Video Call thời gian thực tích hợp ResNet50 & thuật toán gộp khung hình (Sliding Window & Strict Majority Voting):

```bash
streamlit run app_resnet50_webrtc.py --server.enableCORS=false --server.enableXsrfProtection=false
```

Chạy ứng dụng kiểm tra ảnh tĩnh và video tải lên bằng ResNet50:

```bash
streamlit run app_resnet50.py --server.enableCORS=false --server.enableXsrfProtection=false
```

Chạy các phiên bản ứng dụng giao diện nguyên mẫu cũ:

```bash
streamlit run app_videocall.py --server.enableCORS=false --server.enableXsrfProtection=false
```

```bash
streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false
```
