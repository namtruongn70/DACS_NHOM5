# Báo Cáo Tiến Độ Tuần 7

- **Đề tài:** Hệ thống phát hiện DeepFake chân dung tích hợp DeepLearning
- **Thành viên thực hiện:** Nguyễn Nam Trường & Ngô Nguyễn Việt Thắng

---

## 1. Phân Công Nhiệm Vụ Trong Tuần

| Thành viên | Trọng tâm thực hiện | Công việc cụ thể |
| :--- | :--- | :--- |
| **Nguyễn Nam Trường** | Huấn luyện mô hình mới, tối ưu WebRTC & phát hiện ảnh tĩnh | Huấn luyện **ConvNeXt-Tiny**, đánh giá trên tập test 23.016 mẫu, đối chiếu 3 mô hình (`ConvNeXt-Tiny` → `MobileNetV2` → `ResNet50`), tinh chỉnh bộ lọc chống nháy/cửa sổ trượt cho `ResNet50` và tích hợp cơ chế phát hiện ảnh tĩnh (`STATIC IMAGE DETECTED`). |
| **Ngô Nguyễn Việt Thắng** | Kiểm thử ứng dụng, đánh giá luồng thực tế & tổng hợp đối chiếu | Kiểm thử suy luận trên ảnh/video tĩnh và các bản WebRTC, đối chiếu hành vi `REAL / FAKE` qua luồng nén video call, kiểm thử kịch bản ảnh tĩnh và tổng hợp số liệu/nhận xét cho báo cáo. |

---

## 2. Chi Tiết Công Việc Đã Thực Hiện Trong Tuần

Trong tuần này, nhóm tập trung huấn luyện và đánh giá mô hình mới **ConvNeXt-Tiny**, kiểm thử trên cả ảnh/video tĩnh và luồng video call trực tiếp qua **WebRTC**, đối chiếu kết quả trên tập test (23.016 mẫu gồm 11.292 mẫu Fake và 11.724 mẫu Real) theo thứ tự **Model mới (`ConvNeXt-Tiny`) → `MobileNetV2` → `ResNet50`**, đồng thời hoàn thiện bộ lọc chống nháy, cân bằng độ nhạy và bổ sung cơ chế phát hiện đối phương sử dụng **ảnh tĩnh** trên luồng thời gian thực.

### A. Huấn luyện & đánh giá mô hình mới (`ConvNeXt-Tiny`)
<img width="1026" height="633" alt="{D9B7491F-44D1-47CC-8240-B2511AE2B6D5}" src="https://github.com/user-attachments/assets/04aca1fa-c47b-4ec9-a03e-a7882d375dd7" />

- **Huấn luyện & lưu trọng số:** Huấn luyện mô hình **ConvNeXt-Tiny** (`best_convnext_tiny_deepfake.pth`) kèm các tệp cấu hình/chỉ số hỗ trợ (`convnext_tiny_meta.json`, `class_indices_convnext.json`).
  <img width="759" height="514" alt="{863FF29A-982A-447B-BDDD-F0E803B9A9D9}" src="https://github.com/user-attachments/assets/00636210-9d7c-4d1c-9804-1cc6f4169939" />

- **Đánh giá trên tập test (23.016 mẫu):** Đạt **Test Accuracy 99,24%** tại ngưỡng `th=0.43`, phân loại đúng **11.119** mẫu Fake và **11.721** mẫu Real; chỉ lọt **173** mẫu Fake (dự đoán nhầm thành Real) và báo nhầm **3** mẫu Real thành Fake (kèm lưu ảnh ma trận nhầm lẫn vào `confusion_matrix_test.png`).
- **Kiểm thử suy luận ảnh/video tĩnh:** Triển khai ứng dụng phân tích ảnh và video tĩnh (`app_convnext_img_vid.py`), xác nhận mô hình hoạt động rất tốt trên dữ liệu tĩnh có độ nét và phân bố ổn định.

### B. Triển khai, chẩn đoán & tinh chỉnh trên luồng Video Call WebRTC

- **Phát triển nhiều phiên bản ứng dụng WebRTC:** Xây dựng và kiểm thử các bản `app_convnext_webrtc_final.py`, `app_convnext_webrtc_nofjson.py`, `app_convnext_webrtc_high_th.py` và `app_convnext_webrtc_tuned.py` trên `streamlit-webrtc` kết hợp dò mặt OpenCV (Haar Cascade/YuNet) và chuẩn hóa ImageNet.
- **Phân tích nguyên nhân `ConvNeXt-Tiny` dự đoán real-time kém hơn dù chỉ số test rất cao:**
  - **Lệch miền dữ liệu (Domain Shift):** Tập huấn luyện và test gồm ảnh/video tĩnh có chi tiết vi mô rõ nét, trong khi luồng video call WebRTC chịu nén khung hình, sụt bitrate, đổi độ phân giải và biến động ánh sáng làm biến dạng hoặc mất đi các đặc trưng vi mô mà mô hình bám vào.
  - **Bão hòa xác suất một chiều:** Trên luồng nén của video call, mô hình dễ đẩy xác suất dồn hẳn về một phía (`RawReal` sát `1.00`, `FakeAvg` / `FakeMax` gần `0.00`), dẫn đến báo `REAL` liên tục hoặc bị ép thành `FAKE ~100%` đồng loạt khi bật đảo chiều xác suất (`INVERT_PROB = True`).
  - **Áp lực thời gian thực & nhòe chuyển động:** Khuôn mặt chuyển động liên tục, quay đầu hoặc bị `motion blur` kết hợp độ trễ suy luận (~200 ms/frame ở cấu hình chưa tối ưu) khiến phản hồi từng khung hình kém ổn định hơn trên luồng trực tiếp.
- **Thử nghiệm ngưỡng, cửa sổ trượt & thống kê khung hình:** Bổ sung điều chỉnh ngưỡng, kích thước cửa sổ frame, `INFERENCE_STRIDE`, `FP16` và các chỉ số theo dõi (`FakeAvg`, `FakeMax`, `RealAvg`, `RealMin`) để phân tích độ ổn định trên từng khung hình và chuỗi khung hình.

### C. Đối chiếu 3 mô hình & hoàn thiện bộ lọc ổn định cho `ResNet50`

- **So sánh kết quả test 3 mô hình (theo thứ tự: Model mới → MobileNetV2 → ResNet50):**
  <img width="958" height="409" alt="{69F744CA-D5AB-4460-B903-BE385465BA6E}" src="https://github.com/user-attachments/assets/fc3af577-dca5-425e-bc08-f24babff6486" />

  - **ConvNeXt-Tiny (model mới):** **99,24%** accuracy (`th=0.43`) — 11.119 đúng Fake, 11.721 đúng Real, 173 lọt Fake, 3 báo nhầm Real.
  - **MobileNetV2:** **88,53%** accuracy — 9.969 đúng Fake, 10.406 đúng Real, 1.323 lọt Fake, 1.318 báo nhầm Real.
  - **ResNet50:** **98,24%** accuracy — 10.986 đúng Fake, 11.624 đúng Real, 306 lọt Fake, 100 báo nhầm Real.
- **Tinh chỉnh bộ lọc chống nháy & cân bằng độ nhạy cho `ResNet50`:** Khắc phục tình trạng thông báo `NO FACE DETECTED` / `MOVING TOO FAST` nhấp nháy bằng cơ chế giữ khung mặt/trạng thái ngắn hạn (`MAX_HOLD_FRAMES`), chỉnh ngưỡng lọc mờ (`BLUR_VAR_THRESHOLD`) và tinh chỉnh cửa sổ trượt (`FRAME_WINDOW_SIZE`, `DECISION_THRESHOLD`, `FAST_FAKE_THRESHOLD`, `MIN_FAKE_RATIO`) nhằm giảm báo sai khi đang `REAL` mà vẫn bắt kịp khi chuyển sang `FAKE`.

### D. Bổ sung cơ chế phát hiện đối phương sử dụng ảnh tĩnh (Static Image Detection)

<img width="979" height="790" alt="{A7DEA60D-E954-45E9-958A-11427341820A}" src="https://github.com/user-attachments/assets/d5d00b65-ddaf-4ad1-851f-7b0914d5bfd4" />

- **Vấn đề thực tế:** Trong tình huống gọi video, đối phương có thể không sử dụng công nghệ deepfake chuyển động mà dùng **ảnh chân dung người thật tĩnh** (đưa ảnh trước camera hoặc chia sẻ màn hình một bức ảnh cố định). Nếu chỉ dựa vào mô hình phân loại từng khung hình, ảnh người thật này vẫn có thể được phân loại là `REAL`.
- **Giải pháp triển khai:**
  - **Theo dõi sai khác vi chuyển động giữa các khung hình:** Tính độ sai khác cường độ điểm ảnh trung bình trên ảnh xám thu nhỏ giữa frame hiện tại và frame liền trước (`raw_diff`). Khi người thật gọi trực tiếp, luôn tồn tại các vi chuyển động tự nhiên (chớp mắt, cử động cơ mặt, rung nhẹ đầu và nhiễu cảm biến camera); ngược lại, khi đối phương phát ảnh tĩnh, độ sai khác này duy trì ở mức cực thấp.
  - **Làm mượt EMA & giữ trạng thái chống nhấp nháy:** Kết hợp trung bình trượt hàm mũ (`EMA`) cùng bộ đếm khung hình đạt ngưỡng liên tiếp (`STATIC_MIN_FRAMES`) và cơ chế giữ trạng thái ngắn hạn (`STATIC_HOLD_FRAMES`) để hiển thị cảnh báo **`STATIC IMAGE DETECTED`** ổn định, tránh nhấp nháy khi luồng WebRTC có nhiễu nén nhẹ, đồng thời tách biệt hoàn toàn để không ảnh hưởng đến tham số nhận diện `REAL / FAKE` của `ResNet50`.

---

## 3. Bảng So Sánh Chi Tiết (Model Mới → MobileNetV2 → ResNet50)

| Tiêu chí | Model mới: ConvNeXt-Tiny | MobileNetV2 | ResNet50 |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **99,24%** | **88,53%** | **98,24%** |
| **Đúng Fake / Lọt Fake (Fake → Real)** | 11.119 đúng / **173** báo nhầm | 9.969 đúng / **1.323** báo nhầm | 10.986 đúng / **306** báo nhầm |
| **Đúng Real / Báo nhầm Fake (Real → Fake)** | 11.721 đúng / **3** báo nhầm | 10.406 đúng / **1.318** báo nhầm | 11.624 đúng / **100** báo nhầm |
| **Ảnh / video tĩnh** | Rất cao, ít sai số nhất trên tập test | Khá, tỷ lệ nhầm lẫn còn cao trên cả hai lớp | Cao, ổn định, sai số thấp hơn nhiều so với MobileNetV2 |
| **Video call real-time (WebRTC)** | Chỉ số test rất cao nhưng dễ lệch miền/bão hòa xác suất khi qua luồng nén video call thực tế | Nhẹ, FPS cao, nhưng độ phân biệt thật/giả yếu hơn | Ổn định tốt trên luồng WebRTC khi kết hợp bộ lọc cửa sổ trượt, chống nháy mặt & phát hiện ảnh tĩnh |
| **Tốc độ & tài nguyên** | Nặng hơn khi suy luận liên tục từng frame trên luồng trực tiếp | Rất nhẹ, tối ưu cho thiết bị cấu hình thấp | Cân bằng tốt giữa hiệu năng và độ trễ khi bật tối ưu suy luận |

---

## 4. Lựa Chọn Mô Hình Theo Cấu Hình Máy, Hiệu Năng Máy Và Hiệu Năng Dự Đoán

- **Cấu hình & hiệu năng máy:** Chạy trên máy tính cá nhân cấu hình phổ thông / GPU tầm thấp (như NVIDIA GeForce MX350) nên tài nguyên có hạn; mô hình nặng chạy liên tục từng frame dễ tăng độ trễ (~200 ms/frame), còn mô hình nhẹ giữ tốc độ cao nhưng trích xuất đặc trưng sâu kém hơn.
- **Chọn `ResNet50` làm mô hình chính cho WebRTC:** Hiệu năng dự đoán trên tập test rất cao (**98,24%**; 10.986 đúng Fake, 11.624 đúng Real, 306 lọt Fake, 100 báo nhầm Real) và cân bằng tốt giữa độ chính xác, tốc độ xử lý và độ ổn định khi kết hợp bộ lọc cửa sổ trượt, chống nháy mặt và cảnh báo ảnh tĩnh trên máy hiện tại.
- **`ConvNeXt-Tiny` (model mới):** Hiệu năng dự đoán cao nhất trên tập test (**99,24%**; 11.119 đúng Fake, 11.721 đúng Real, 173 lọt Fake, 3 báo nhầm Real), nhưng nặng hơn khi chạy liên tục và dễ lệch miền/bão hòa xác suất trên luồng WebRTC nén — phù hợp cho **phân tích ảnh/video tĩnh** hoặc **fine-tune thêm bằng dữ liệu thu từ luồng call**.
- **`MobileNetV2`:** Đạt **88,53%** (9.969 đúng Fake, 10.406 đúng Real, 1.323 lọt Fake, 1.318 báo nhầm Real) — chỉ giữ làm **baseline đối chiếu** hoặc dự phòng khi cần tốc độ cao trên máy rất yếu.

---

## 5. Kế Hoạch Tuần Tới

1. **Chốt cấu hình WebRTC chính thức với `ResNet50`:** Kiểm thử thực tế luân phiên `REAL`, `FAKE` và kịch bản dùng ảnh tĩnh để chốt ngưỡng và bộ lọc mượt mà nhất cho buổi demo.
2. **Tổng hợp bộ số liệu thực nghiệm:** Ghi nhận đầy đủ chỉ số test (`99,24%`, `88,53%`, `98,24%`) kết hợp `FPS`, `Latency` và nhận xét các tình huống biên (mất mặt, chuyển động nhanh, ánh sáng thay đổi, dùng ảnh tĩnh).
