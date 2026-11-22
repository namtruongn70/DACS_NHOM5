## BÁO CÁO TIẾN ĐỘ TUẦN 6

- **Đề tài:** Hệ thống phát hiện Deepfake chân dung tích hợp Deep Learning (`app_resnet50_webrtc.py`)
- **Thời gian:** Tuần 6 (Tháng 11/2026)

---

### 1. Phân công nhiệm vụ chi tiết (Tuần 6)

| Thành viên | Trọng tâm thực hiện | Công việc cụ thể |
| :--- | :--- | :--- |
| **Nguyễn Nam Trường** | Tối ưu luồng xử lý & Đo hiệu năng | Cập nhật `app_resnet50_webrtc.py` (downscale dò mặt, bỏ `PIL.Image`, cấu hình `FP16`/stride, luồng `640x480`); đo đạc thông số trước và sau tối ưu trên máy cá nhân (GPU MX350) và tổng hợp báo cáo. |
| **Ngô Nguyễn Việt Thắng** | Kiểm thử thực nghiệm & Đánh giá độ mượt | Kiểm thử luồng Sender/Receiver trước và sau tối ưu; đối chiếu độ ổn định nhãn `REAL/FAKE`, phản ứng cảnh báo (`NO FACE DETECTED`, `WARNING: MOVING TOO FAST`) và ghi nhận đánh giá thực tế. |

---

### 2. Tóm tắt kết quả đạt được trong tuần

Trong tuần 6, nhóm tập trung tích hợp công cụ đo hiệu năng trực tiếp trên giao diện Video Call WebRTC và tối ưu luồng xử lý khung hình trên máy cá nhân sử dụng GPU NVIDIA GeForce MX350. Kết quả đo đạc thực tế qua hai trạng thái **trước tối ưu** và **sau tối ưu** cho thấy tốc độ xử lý tăng gần gấp đôi và độ trễ giảm gần một nửa, trong khi nhãn nhận diện vẫn giữ sự ổn định.

---

### 3. So sánh kết quả thực nghiệm: Trước tối ưu và Sau tối ưu

#### 3.1. Các giải pháp tối ưu đã triển khai

- **Dò khuôn mặt trên ảnh thu nhỏ (`DETECT_SCALE = 0.5`):** Thu nhỏ ảnh xám trước khi chạy `Haar Cascade` để giảm tải CPU, sau đó ánh xạ tọa độ ngược lại ảnh gốc nhằm giữ nguyên độ phân giải vùng mặt đưa vào mô hình.
- **Bỏ bước chuyển đổi qua `PIL.Image`:** Chuyển trực tiếp mảng NumPy (`RGB`) sang `torch.Tensor` (`224x224`) và chuẩn hóa trực tiếp bằng tensor.
- **Tối ưu suy luận và luồng WebRTC:** Áp dụng suy luận `float16` trên CUDA (`USE_FP16_CUDA`), hỗ trợ bước nhảy khung hình (`INFERENCE_STRIDE`) và giới hạn độ phân giải luồng WebRTC ở mức `640x480` để giảm nghẽn khung hình.

#### 3.2. Bảng đối chiếu thông số đo đạc (`W = 9`)

| Chỉ tiêu đo | Trước tối ưu (`image_9a99a1.jpg`) | Sau tối ưu (`image_9a99de.jpg`, `S = 1`) | Mức thay đổi |
| :--- | :---: | :---: | :--- |
| **Proc FPS** | 14.5 | 26.5 | Tăng +12.0 FPS (gần gấp đôi) |
| **Stream FPS** | 13.8 | 20.1 | Tăng +6.3 FPS (luồng mượt hơn rõ rệt) |
| **Latency (Lat)** | 69.0 ms | 37.7 ms | Giảm 31.3 ms (giảm gần một nửa độ trễ) |
| **Nhãn & Độ tự tin** | REAL (82.4%) | REAL (81.2%) | Ổn định ở nhãn REAL, khung mặt bám sát |

#### 3.3. Nhận xét tổng quan
<img width="850" height="625" alt="image" src="https://github.com/user-attachments/assets/6d2d0ce5-2905-49e4-9b8d-cd63aa6b568a" />

- **Trước tối ưu:** Hệ thống đạt **Proc FPS: 14.5**, **Stream FPS: 13.8** và **Latency: 69.0 ms** với cửa sổ biểu quyết `W = 9`, nhận diện chính xác **REAL (82.4%)**.
- <img width="851" height="692" alt="image" src="https://github.com/user-attachments/assets/04b4647e-7455-4a57-919c-ea468131f3f7" />

- **Sau tối ưu:** Khi giữ nguyên `W = 9` và chạy suy luận trên mọi khung hình có mặt (`S = 1`), hệ thống đạt **Proc FPS: 26.5**, **Stream FPS: 20.1** và **Latency: 37.7 ms**, đồng thời giữ ổn định nhãn **REAL (81.2%)** mà không làm giảm chất lượng phân loại.

---

### 4. Hướng triển khai tiếp theo (Tuần 7)

- Huấn luyện thêm các mô hình khác (ví dụ các kiến trúc nhẹ hoặc biến thể CNN/backbone phù hợp) để so sánh toàn diện giữa các mô hình về độ chính xác, tốc độ suy luận và mức tiêu tốn tài nguyên.
- Đối chiếu và lựa chọn mô hình phù hợp chính thức với điều kiện thực tế của hệ thống như cấu hình phần cứng (GPU/CPU/RAM) và chi phí triển khai.
- Tiếp tục đo lặp lại nhiều lần (30–60 giây mỗi phiên), thử nghiệm thêm `INFERENCE_STRIDE = 2` và các mức `FRAME_WINDOW_SIZE` (`5`, `9`, `15`) trên cả video thật và video Deepfake để hoàn thiện báo cáo.
