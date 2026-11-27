# Báo Cáo Tiến Độ Tuần (21/11/2026 – 27/11/2026)

- **Dự án:** Hệ thống giám sát & phát hiện Deepfake thời gian thực qua Video Call (WebRTC)
- **Thành viên thực hiện:** Nguyễn Nam Trường & Ngô Nguyễn Việt Thắng

---

## 1. Chi Tiết Công Việc Đã Thực Hiện Trong Tuần

Trong tuần này, nhóm tập trung huấn luyện và đánh giá mô hình mới **ConvNeXt-Tiny**, kiểm thử trên cả ảnh/video tĩnh và luồng video call trực tiếp qua **WebRTC**, đối chiếu kết quả trên tập test (23.016 mẫu gồm 11.292 mẫu Fake và 11.724 mẫu Real) theo thứ tự **Model mới (`ConvNeXt-Tiny`) → `MobileNetV2` → `ResNet50`**, đồng thời hoàn thiện bộ lọc chống nháy và cân bằng độ nhạy cho luồng thời gian thực.

### A. Huấn luyện & đánh giá mô hình mới (`ConvNeXt-Tiny`)

  <img width="922" height="576" alt="{4AF1E7F4-38D6-449B-B055-815B4BD02DAA}" src="https://github.com/user-attachments/assets/799ff323-43c4-41c9-9f67-a53e041dd342" />

- **Huấn luyện & lưu trọng số:** Huấn luyện mô hình **ConvNeXt-Tiny** (`best_convnext_tiny_deepfake.pth`) kèm các tệp cấu hình/chỉ số hỗ trợ (`convnext_tiny_meta.json`, `class_indices_convnext.json`).
<img width="808" height="517" alt="image" src="https://github.com/user-attachments/assets/468e6ba4-3ff4-43e7-9a10-7804d16be9d3" />

- **Đánh giá trên tập test (23.016 mẫu):** Đạt **Test Accuracy 99,24%** tại ngưỡng `th=0.43`, phân loại đúng **11.119** mẫu Fake và **11.721** mẫu Real; chỉ lọt **173** mẫu Fake (dự đoán nhầm thành Real) và báo nhầm **3** mẫu Real thành Fake (kèm lưu ảnh ma trận nhầm lẫn vào `confusion_matrix_test.png`).
- **Kiểm thử suy luận ảnh/video tĩnh:** Triển khai ứng dụng phân tích ảnh và video tĩnh (`app_convnext_img_vid.py`), xác nhận mô hình hoạt động rất tốt trên dữ liệu tĩnh có độ nét và phân bố ổn định[cite: 2].

### B. Triển khai, chẩn đoán & tinh chỉnh trên luồng Video Call WebRTC

- **Phát triển nhiều phiên bản ứng dụng WebRTC:** Xây dựng và kiểm thử các bản `app_convnext_webrtc_final.py`, `app_convnext_webrtc_nofjson.py`, `app_convnext_webrtc_high_th.py` và `app_convnext_webrtc_tuned.py` trên `streamlit-webrtc` kết hợp dò mặt OpenCV (Haar Cascade/YuNet) và chuẩn hóa ImageNet.
- **Phân tích nguyên nhân `ConvNeXt-Tiny` dự đoán real-time kém hơn dù chỉ số test rất cao:**
  - **Lệch miền dữ liệu (Domain Shift):** Tập huấn luyện và test gồm ảnh/video tĩnh có chi tiết vi mô rõ nét, trong khi luồng video call WebRTC chịu nén khung hình, sụt bitrate, đổi độ phân giải và biến động ánh sáng làm biến dạng hoặc mất đi các đặc trưng vi mô mà mô hình bám vào.
  - **Bão hòa xác suất một chiều:** Trên luồng nén của video call, mô hình dễ đẩy xác suất dồn hẳn về một phía (`RawReal` sát `1.00`, `FakeAvg` / `FakeMax` gần `0.00`), dẫn đến báo `REAL` liên tục hoặc bị ép thành `FAKE ~100%` đồng loạt khi bật đảo chiều xác suất (`INVERT_PROB = True`).
  - **Áp lực thời gian thực & nhòe chuyển động:** Khuôn mặt chuyển động liên tục, quay đầu hoặc bị `motion blur` kết hợp độ trễ suy luận (~200 ms/frame ở cấu hình chưa tối ưu) khiến phản hồi từng khung hình kém ổn định hơn trên luồng trực tiếp.
- **Thử nghiệm ngưỡng, cửa sổ trượt & thống kê khung hình:** Bổ sung điều chỉnh ngưỡng, kích thước cửa sổ frame, `INFERENCE_STRIDE`, `FP16` và các chỉ số theo dõi (`FakeAvg`, `FakeMax`, `RealAvg`, `RealMin`) để phân tích độ ổn định trên từng khung hình và chuỗi khung hình.

### C. Đối chiếu 3 mô hình & hoàn thiện bộ lọc ổn định cho `ResNet50`

- **So sánh kết quả test 3 mô hình (theo thứ tự: Model mới → MobileNetV2 → ResNet50):**
  <img width="975" height="417" alt="image" src="https://github.com/user-attachments/assets/695332ad-b199-4e68-bcca-5a19b7f00bd6" />

  - **ConvNeXt-Tiny (model mới):** **99,24%** accuracy (`th=0.43`) — 11.119 đúng Fake, 11.721 đúng Real, 173 lọt Fake, 3 báo nhầm Real[cite: 2].
  - **MobileNetV2:** **88,53%** accuracy — 9.969 đúng Fake, 10.406 đúng Real, 1.323 lọt Fake, 1.318 báo nhầm Real[cite: 3].
  - **ResNet50:** **98,24%** accuracy — 10.986 đúng Fake, 11.624 đúng Real, 306 lọt Fake, 100 báo nhầm Real[cite: 4].
- **Tinh chỉnh bộ lọc chống nháy & cân bằng độ nhạy cho `ResNet50`:** Khắc phục tình trạng thông báo `NO FACE DETECTED` / `MOVING TOO FAST` nhấp nháy bằng cơ chế giữ khung mặt/trạng thái ngắn hạn (`MAX_HOLD_FRAMES`), chỉnh ngưỡng lọc mờ (`BLUR_VAR_THRESHOLD`) và tinh chỉnh cửa sổ trượt (`FRAME_WINDOW_SIZE`, `DECISION_THRESHOLD`, `FAST_FAKE_THRESHOLD`, `MIN_FAKE_RATIO`) nhằm giảm báo sai khi đang `REAL` mà vẫn bắt kịp khi chuyển sang `FAKE`.

---

## 2. Bảng So Sánh Chi Tiết (Model Mới → MobileNetV2 → ResNet50)

| Tiêu chí | Model mới: **ConvNeXt-Tiny** | **MobileNetV2** | **ResNet50** |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **99,24%**  | **88,53%** | **98,24%** |
| **Đúng Fake / Lọt Fake (Fake → Real)** | 11.119 đúng / **173** báo nhầm | 9.969 đúng / **1.323** báo nhầm | 10.986 đúng / **306** báo nhầm |
| **Đúng Real / Báo nhầm Fake (Real → Fake)** | 11.721 đúng / **3** báo nhầm | 10.406 đúng / **1.318** báo nhầm | 11.624 đúng / **100** báo nhầm |
| **Ảnh / video tĩnh** | Rất cao, ít sai số nhất trên tập test | Khá, tỷ lệ nhầm lẫn còn cao trên cả hai lớp | Cao, ổn định, sai số thấp hơn nhiều so với MobileNetV2 |
| **Video call real-time (WebRTC)** | Chỉ số test rất cao nhưng dễ lệch miền/bão hòa xác suất khi qua luồng nén video call thực tế | Nhẹ, FPS cao, nhưng độ phân biệt thật/giả yếu hơn | Ổn định tốt trên luồng WebRTC khi kết hợp bộ lọc cửa sổ trượt & chống nháy mặt |
| **Tốc độ & tài nguyên** | Nặng hơn khi suy luận liên tục từng frame trên luồng trực tiếp | Rất nhẹ, tối ưu cho thiết bị cấu hình thấp | Cân bằng tốt giữa hiệu năng và độ trễ khi bật tối ưu suy luận |

---

## 3. Lựa Chọn Mô Hình Theo Cấu Hình Máy, Hiệu Năng Máy Và Hiệu Năng Dự Đoán

- **Cấu hình & hiệu năng máy:** Chạy trên máy tính cá nhân cấu hình phổ thông / GPU tầm thấp (như NVIDIA GeForce MX350) nên tài nguyên có hạn; mô hình nặng chạy liên tục từng frame dễ tăng độ trễ (~200 ms/frame), còn mô hình nhẹ giữ tốc độ cao nhưng trích xuất đặc trưng sâu kém hơn.
- **Chọn `ResNet50` làm mô hình chính cho WebRTC:** Hiệu năng dự đoán trên tập test rất cao (**98,24%**; 10.986 đúng Fake, 11.624 đúng Real, 306 lọt Fake, 100 báo nhầm Real) và cân bằng tốt giữa độ chính xác, tốc độ xử lý và độ ổn định khi kết hợp bộ lọc cửa sổ trượt + chống nháy mặt trên máy hiện tại.
- **`ConvNeXt-Tiny` (model mới):** Hiệu năng dự đoán cao nhất trên tập test (**99,24%**; 11.119 đúng Fake, 11.721 đúng Real, 173 lọt Fake, 3 báo nhầm Real), nhưng nặng hơn khi chạy liên tục và dễ lệch miền/bão hòa xác suất trên luồng WebRTC nén — phù hợp cho **phân tích ảnh/video tĩnh** hoặc **fine-tune thêm bằng dữ liệu thu từ luồng call**.
- **`MobileNetV2`:** Đạt **88,53%** (9.969 đúng Fake, 10.406 đúng Real, 1.323 lọt Fake, 1.318 báo nhầm Real) — chỉ giữ làm **baseline đối chiếu** hoặc dự phòng khi cần tốc độ cao trên máy rất yếu[cite: 2, 3, 4].

---

## 4. Kế Hoạch Tuần Tới

1. **Chốt cấu hình WebRTC chính thức với `ResNet50`:** Kiểm thử thực tế luân phiên `REAL` và `FAKE` để chốt ngưỡng và bộ lọc mượt mà nhất cho buổi demo.
2. **Tổng hợp bộ số liệu thực nghiệm:** Ghi nhận đầy đủ chỉ số test (`99,24%`, `88,53%`, `98,24%`) kết hợp `FPS`, `Latency` và nhận xét các tình huống biên (mất mặt, chuyển động nhanh, ánh sáng thay đổi).
