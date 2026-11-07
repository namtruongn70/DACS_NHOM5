#**Tuần 4: Phát triển Giao diện Web (Streamlit), Xử lý File Video & Real-time Webcam**

##**1. Việc đã làm (Dựa trên Mã nguồn Thực tế):**

  **-Xây dựng Giao diện Web Trực quan với Streamlit (app.py):**

  +Thiết lập môi trường chạy ứng dụng web cục bộ thông qua thư viện Streamlit (st.set_page_config), tối ưu hóa hiển thị bố cục gọn gàng, hỗ trợ người dùng tương tác kéo thả tệp dễ dàng.

  +Tích hợp cơ chế cache tài nguyên thông minh (@st.cache_resource) để nạp sẵn mô hình Deepfake (best_deepfake_model.h5 hoặc deepfake_model.h5) và bộ phát hiện khuôn mặt YuNet (face_detection_yunet_2023mar.onnx tự động tải từ kho lưu trữ OpenCV Zoo nếu chưa có), giúp tăng tốc độ khởi động ứng dụng và tránh lãng phí tài nguyên tính toán.
<img width="1601" height="603" alt="{69E84DBD-ABD6-42FF-A430-BC851F1A31CD}" src="https://github.com/user-attachments/assets/f1bf8741-4947-49a3-98b7-cfc5019d7a96" />

  **-Phát triển Module Phân tích Ảnh đơn & Chuỗi Thời gian Video:**
<img width="1580" height="631" alt="{E375CAB5-1086-4075-A4BC-5BF0FF442508}" src="https://github.com/user-attachments/assets/9ae3ca4c-2987-4080-a56d-9641deb784ff" />

  +Xử lý ảnh: Nhận ảnh tải lên qua st.file_uploader, giải mã bằng OpenCV (cv2.imdecode), chuyển đổi không gian màu RGB, gọi hàm extract_face_crop sử dụng YuNet để định vị, cắt sát vùng mặt với tỷ lệ chuẩn và đưa qua MobileNetV2 để dự đoán xác suất Sigmoid, từ đó đưa ra kết quả kết luận REAL hay DEEPFAKE.
  
<img width="409" height="828" alt="{C48F88F1-4B55-4F5F-BBCB-276C64A44F7D}" src="https://github.com/user-attachments/assets/a01b84f0-34a1-4698-862a-0b0564ecca00" />

  +Xử lý video chuyên sâu (FRAME_INTERVAL = 2): Lưu video tạm thời qua tempfile, đọc luồng bằng cv2.VideoCapture, thiết lập cơ chế quét định kỳ (cứ mỗi 2 frame lấy 1 frame để tối ưu hóa thời gian chờ) và tính toán các chỉ số thống kê nâng cao: Điểm trung vị (median), điểm trung bình (mean), tỷ lệ % khung hình nghi vấn Fake (< 0.45), cũng như chuỗi frame Fake liên tiếp dài nhất để đưa ra thuật toán đánh giá tổng hợp toàn bộ video.

 **-Xây dựng Module Nhận diện Real-time qua Video Call (app_videocall.py):**
<img width="1730" height="869" alt="{37C30450-3967-43E0-90CE-4E9562663AD9}" src="https://github.com/user-attachments/assets/e26d6176-745e-4822-a44b-826570bfbdb3" />

  +Tích hợp thư viện streamlit-webrtc để thiết lập kết nối WebRTC ngang hàng (P2P qua cấu hình ICE Server STUN của Google), chia tách rõ ràng hai vai trò riêng biệt: Người phát (Sender / Kẻ tấn công) và Hệ thống giám sát (Receiver / AI Detector) thông qua mã phòng tùy chỉnh (room_id).

  +Xây dựng luồng xử lý khung hình thời gian thực (process_video_frame), trong đó ứng dụng sử dụng bộ đệm trượt deque (FRAME_WINDOW_SIZE = 10) để lưu trữ lịch sử 10 frame gần nhất, kết hợp cơ chế ngưỡng nhãn nghiêm ngặt (FAKE_THRESHOLD = 70.0%) để chống báo sai đối với mặt thật, tự động vẽ khung chữ nhật (Đỏ cho Fake, Xanh lá cho Real) và hiển thị banner cảnh báo trực tiếp trên màn hình video call.

#**2. Khó khăn Gặp phải:**
  -Xung đột luồng và độ phức tạp kiến trúc (Thread Conflict): Việc tích hợp dồn dập nhiều tính năng nặng (vừa xử lý tệp video dung lượng lớn trên Streamlit, vừa mở đồng thời luồng WebRTC thời gian thực đa luồng qua streamlit-webrtc) khiến cấu trúc mã nguồn trở nên phức tạp, dễ xảy ra tranh chấp tài nguyên bộ nhớ hoặc xung đột vòng lặp giữa các phiên bản giao diện.

  -Sụt giảm hiệu năng (FPS thấp) và hiện tượng giật lag: Khi bật camera chạy real-time ở các phiên bản đầu, mô hình phải xử lý liên tục 100% số frame đi qua, kết hợp việc gọi đồng thời YuNet và MobileNetV2 trên mỗi frame khiến CPU/GPU bị quá tải, dẫn đến hiện tượng trễ hình (latency) và tụt tốc độ khung hình nghiêm trọng trên màn hình cuộc gọi.

#**3. Định hướng Tuần tới:**
  -Tiến hành tối ưu hóa hiệu năng hệ thống toàn diện bằng các kỹ thuật giảm tải khung hình chuyên sâu (như cải tiến cơ chế Frame Skipping hoặc điều chỉnh độ phân giải đầu vào WebRTC) nhằm đảm bảo tốc độ khung hình luôn mượt mà trong các thử nghiệm thực tế tiếp theo.
