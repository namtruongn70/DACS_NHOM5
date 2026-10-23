PHẦN 1: BỘ DỮ LIỆU GỐC 140.000 ẢNH (KAGGLE DATASET)

Quy mô tổng thể: 140.000 ảnh được chia hoàn toàn cân bằng: 70.000 ảnh Real và 70.000 ảnh Fake.

Nguồn gốc 70.000 ảnh Real (Ảnh thật):

Được trích xuất từ các kho dữ liệu chân dung tiêu chuẩn quốc tế, chủ yếu là FFHQ (Flickr-Faces-HQ) do NVIDIA phát triển.

Đặc điểm: Chứa khuôn mặt người thật với chất lượng ảnh sắc nét, cực kỳ đa dạng về chủng tộc (Á, Âu, Phi, Mỹ Latinh...), độ tuổi (từ trẻ em đến người già), giới tính, biểu cảm (cười, khóc, bình thường), góc chụp (chính diện, nghiêng), phụ kiện (kính mắt, mũ) và điều kiện ánh sáng tự nhiên khác nhau.

Nguồn gốc 70.000 ảnh Fake (Ảnh giả mạo):

Được tổng hợp từ các bộ dataset sinh ảnh giả lập bằng trí tuệ nhân tạo (TI), tiêu biểu là các mạng sinh đối kháng (GANs như StyleGAN, ProGAN) hoặc các thử thách phát hiện deepfake công cộng.

Đặc điểm: Chứa các dấu vết kỹ thuật số đặc trưng (artifacts) như sự bất cân xứng nhẹ ở mắt/tai, độ mịn da quá mức do AI tạo ra, hoặc các lỗi hòa trộn vùng viền giữa mặt và phông nền mà mắt thường khó nhận thấy.

PHẦN 2: BỘ DỮ LIỆU TỰ THU THẬP (~10.000 ẢNH)

Quy trình xử lý & Trích xuất kỹ thuật:

Mở tool Deep-Live-Cam.

Ở mục Target, chọn file video mẫu của người khác.

Bấm Start để phần mềm tự động thay mặt bạn thành mặt người đó và xuất ra một Video FAKE hoàn chỉnh.

Chọn video qua giao diện: Sử dụng script Python tích hợp Tkinter để mở cửa sổ chọn file video từ ổ đĩa máy tính một cách trực quan.

Phát hiện và Cắt mặt chính xác (YuNet + Tight Crop):

Sử dụng mô hình YuNet ONNX (face_detection_yunet_2023mar.onnx) để quét từng frame video.

Tự động lấy khuôn mặt có độ tin cậy cao nhất, áp dụng vùng đệm mở rộng (padding) khoảng 10-15% để giữ trọn vùng cằm, trán, tai mà không lấy quá nhiều phông nền phía sau.

Chuẩn hóa kích thước: Mọi ảnh cắt ra đều được resize tự động về chuẩn 224 x 224 pixels để tương thích hoàn toàn với đầu vào của MobileNetV2.

Quy tắc đặt tên chuẩn hóa:

Ảnh thật tự thu thập: user_real_<timestamp>_<stt>.jpg

Ảnh giả tự thu thập: user_fake_<timestamp>_<stt>.jpg

Làm sạch tự động (Data Cleaning):

Chạy script chuyên biệt (clean_no_face_user_fake.py) dùng YuNet để quét lại toàn bộ thư mục train/fake (và tương tự cho train/real).

Tự động phát hiện và xóa bỏ vĩnh viễn các file ảnh bị lỗi, ảnh mờ nhòe do chuyển động nhanh (motion blur), hoặc các khung hình không tìm thấy khuôn mặt, đảm bảo 100% dữ liệu đưa vào train đều hợp lệ.
