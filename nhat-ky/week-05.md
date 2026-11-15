# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN - TUẦN 5
* **Nhóm:** Nhóm 5
* **Thành viên:** 
  1. Nguyễn Nam Trường
  2. Ngô Nguyễn Việt Thắng
* **Đề tài:** Hệ thống phát hiện Deepfake chân dung tích hợp Deep Learning.

---

## I. TỔNG KẾT Ý KIẾN ĐÓNG GÓP CỦA GIẢNG VIÊN VÀ HƯỚNG GIẢI QUYẾT

### 1. Ghi nhận từ Thầy
* Thầy đã kiểm tra kho chứa mã nguồn (Repository) của nhóm và xác nhận phần ứng dụng xử lý ảnh, video tĩnh cũng như tính năng gọi video call real-time đã chạy ổn định.

### 2. Vấn đề cần khắc phục và triển khai trong tuần
* **Vấn đề:** Độ chính xác 88.53% trước đó được đo trên tập Validation (tập dùng để lựa chọn mô hình), dẫn đến kết quả chưa phản ánh khách quan năng lực tổng quát hóa thực tế.
* **Giải pháp thực hiện:** Nhóm đã xây dựng và tách biệt một tập kiểm thử độc lập hoàn toàn (Independent Test Set) gồm 23,016 mẫu (bao gồm 11,292 ảnh Fake và 11,724 ảnh Real được tổng hợp từ Kaggle, Deepfake tự tạo bằng công cụ Deep-Live-Cam và cắt từ video chọn lọc trên internet, tuyệt đối không dùng trong quá trình huấn luyện).
* **Bổ sung:** Thực hiện bảng so sánh định lượng giữa mô hình MobileNetV2 và ResNet50 trên cùng tập test độc lập, đồng thời đánh giá chi tiết cơ chế gộp kết quả theo khung hình cho video.

---

## II. PHÂN CÔNG CÔNG VIỆC VÀ KẾT QUẢ THỰC HIỆN CỦA TỪNG THÀNH VIÊN

### 1. Thành viên: Ngô Nguyễn Việt Thắng
* **Nhiệm vụ được giao:** Chạy và đánh giá hiệu năng mô hình MobileNetV2 trên tập kiểm thử độc lập.
* **Kết quả thực hiện trong tuần:**
  * Thiết lập pipeline đánh giá cho mô hình MobileNetV2 trên tập test độc lập gồm 23,016 mẫu.
    <img width="790" height="325" alt="image" src="https://github.com/user-attachments/assets/2cfb2b81-0a9b-4a71-b19f-54bee4c9e222" />

  * Thu nhận và phân tích các chỉ số định lượng thực tế: Mô hình đạt tỷ lệ chính xác tổng thể (Test Accuracy) 88.53% với ngưỡng phân loại tối ưu là 0.85.
  * Xuất và lưu trữ biểu đồ ma trận nhầm lẫn (`confusion_matrix_optimized.png`) phục vụ công tác tổng kết báo cáo.
    <img width="338" height="310" alt="image" src="https://github.com/user-attachments/assets/571ebc9c-49f5-448d-bfd0-57c9978ab95a" />


### 2. Thành viên: Nguyễn Nam Trường
* **Nhiệm vụ được giao:** Huấn luyện mô hình ResNet50 (PyTorch), chạy trên tập kiểm thử độc lập và tối ưu hóa thuật toán gộp kết quả theo khung hình cho ứng dụng Video Call.
* **Kết quả thực hiện chi tiết trong tuần:**
  * **Huấn luyện mô hình ResNet50 (PyTorch) với chiến lược 2 giai đoạn (`train_resnet50.py`):**
    <img width="851" height="263" alt="image" src="https://github.com/user-attachments/assets/d52bf1d5-6f87-4a91-a68e-ee559dad8576" />

    * *Dữ liệu huấn luyện & kiểm định:* Nạp tổng cộng 70,000 ảnh huấn luyện (gồm 35,000 ảnh Fake: 7,467 ảnh tự thu thập + 27,533 ảnh bổ sung; và 35,000 ảnh Real: 4,843 ảnh tự thu thập + 30,157 ảnh bổ sung) cùng 12,000 ảnh Validation (6,000 Fake và 6,000 Real) trên GPU NVIDIA GeForce MX350.
      <img width="795" height="412" alt="image" src="https://github.com/user-attachments/assets/9f0a7d40-11e3-468a-8a3f-ea7cef9ca9ac" />

    * *Giai đoạn 1 (Warm-up Classifier - 3 Epochs):* Đóng băng các lớp trích xuất đặc trưng gốc để huấn luyện bộ phân loại đầu ra, độ chính xác tập huấn luyện tăng dần từ 81.80% (Epoch 1) lên 85.95% (Epoch 3).
      <img width="901" height="110" alt="image" src="https://github.com/user-attachments/assets/ffb35659-2e9d-49fd-a77d-8db930182792" />

    * *Giai đoạn 2 (Fine-tuning 40 lớp cuối ResNet50):* Mở khóa 40 lớp cuối để tinh chỉnh đặc trưng chuyên sâu, đạt độ chính xác trên tập kiểm định (Validation Accuracy) lên tới 99.49% (Val Loss: 0.0179) và lưu lại bộ trọng số tốt nhất vào file `best_resnet50_model_torch.pth`.
  * **Đánh giá khách quan trên tập kiểm thử độc lập (`eval_test_pytorch.py`):**
    <img width="850" height="245" alt="image" src="https://github.com/user-attachments/assets/79e91150-26b4-43dd-8624-e24aab999a94" />

    * Chạy kiểm thử trên 23,016 ảnh độc lập (11,292 ảnh Fake và 11,724 ảnh Real hoàn toàn tách biệt với dữ liệu huấn luyện).
      <img width="832" height="418" alt="image" src="https://github.com/user-attachments/assets/f0a32c61-1b9b-4391-9750-be65ec760d85" />

    * Mô hình đạt tỷ lệ chính xác tổng thể (Test Accuracy) là 98.24% tại ngưỡng phân loại tối ưu 0.53 (với lớp Fake đạt Precision: 0.9910, Recall: 0.9729, F1-score: 0.9819; lớp Real đạt Precision: 0.9744, Recall: 0.9915, F1-score: 0.9828; ma trận nhầm lẫn nhận diện đúng 10,986 mẫu Fake và 11,624 mẫu Real) và xuất biểu đồ `confusion_matrix_pytorch_test.png`.
      <img width="792" height="517" alt="image" src="https://github.com/user-attachments/assets/9ed1cedd-30ca-4de1-92d6-8bc8caf8990f" />

  * **Tối ưu hóa thuật toán gộp khung hình và triển khai ứng dụng (`app_resnet50.py`, `app_resnet50_webrtc.py`):**
    <img width="975" height="522" alt="image" src="https://github.com/user-attachments/assets/ce10be83-5d32-43c4-9434-c91c42622774" />

    * Tích hợp bộ trọng số `best_resnet50_model_torch.pth` vào cả giao diện kiểm tra ảnh/video tĩnh (`app_resnet50.py`) và hệ thống giám sát Video Call thời gian thực (`app_resnet50_webrtc.py`).
    * Nghiên cứu và cài đặt cơ chế gộp kết quả theo khung hình (Sliding Window & Strict Majority Voting) với cửa sổ khung hình và ngưỡng đồng thuận, kết hợp cân bằng sáng tự động (Histogram Equalization) và bộ lọc trạng thái (`NO FACE DETECTED`, chống nhòe chuyển động) giúp khắc phục triệt để hiện tượng nhấp nháy nhãn khi gọi video thực tế.

---

## III. BẢNG SO SÁNH ĐỊNH LƯỢNG HAI MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP (23,016 MẪU)

| Tiêu chí đánh giá | Mô hình MobileNetV2 | Mô hình ResNet50 (PyTorch) |
| :--- | :--- | :--- |
| **Ngưỡng phân loại tối ưu (Threshold)** | 0.85 | 0.53 |
| **Tỷ lệ chính xác tổng thể (Test Accuracy)** | 88.53% | 98.24% |
| **Chỉ số lớp Giả mạo (Fake Class)** | Precision: 0.8832<br>Recall: 0.8828<br>F1-score: 0.8830 | Precision: 0.9910<br>Recall: 0.9729<br>F1-score: 0.9819 |
| **Chỉ số lớp Thật (Real Class)** | Precision: 0.8872<br>Recall: 0.8876<br>F1-score: 0.8874 | Precision: 0.9744<br>Recall: 0.9915<br>F1-score: 0.9828 |
| **Ma trận nhầm lẫn (Confusion Matrix)** | <img width="338" height="310" alt="image" src="https://github.com/user-attachments/assets/631e9a00-cf41-4885-acad-9123011413c1" />| <img width="357" height="314" alt="image" src="https://github.com/user-attachments/assets/fb80d649-00b4-4772-902b-16ee945ae6d4" />
 |

---

## IV. ĐÁNH GIÁ CƠ CHẾ GỘP KẾT QUẢ THEO KHUNG HÌNH (FRAME-LEVEL AGGREGATION) TRONG VIDEO
* **Vấn đề thực tế:** Việc phân loại độc lập từng khung hình (frame-by-frame inference) trong môi trường video call trực tuyến thường gặp hiện tượng dao động xác suất do thay đổi góc độ chiếu sáng hoặc cử động tự nhiên của người dùng, dẫn đến hiện tượng chớp nháy nhãn kết quả (Real/Fake liên tục).
* **Giải pháp tối ưu đã triển khai:**
  1. **Hàng đợi cửa sổ trượt (Sliding Window Queue):** Lưu trữ lịch sử dự đoán xác suất của các khung hình liên tiếp gần nhất.
  2. **Biểu quyết đa số nghiêm ngặt (Strict Majority Voting):** Yêu cầu tỷ lệ phiếu đồng thuận phải đạt từ 70% trở lên trong cửa sổ trượt thì hệ thống mới chuyển đổi trạng thái hiển thị, giúp khung hình giữ được độ ổn định cao.
  3. **Bộ lọc trạng thái an toàn:** Tự động vô hiệu hóa việc đưa ra kết luận Real/Fake khi camera không tìm thấy khuôn mặt (`NO FACE DETECTED`) hoặc khi hình ảnh bị nhòe do chuyển động lắc đầu nhanh (`Motion Blur`), đảm bảo tính trực quan và độ tin cậy tuyệt đối cho hệ thống giám sát.

---

## V. KẾ HOẠCH CÔNG VIỆC TRONG TUẦN TỚI (TUẦN 6)
* **Tối ưu tốc độ trễ (Latency) Video Call:** Tinh chỉnh kích thước bộ đệm khung hình và tối ưu luồng xử lý của OpenCV kết hợp CUDA GPU trong `app_resnet50_webrtc.py` nhằm tăng tốc độ khung hình (FPS), giúp cuộc gọi mượt mà và giảm độ trễ thực tế.
