import os
import cv2
import numpy as np
import tempfile
import urllib.request
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. CẤU HÌNH MÔI TRƯỜNG & STREAMLIT
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(
    page_title="Hệ Thống Nhận Diện Deepfake", 
    page_icon="🛡️", 
    layout="centered"
)

st.title("🛡️ Hệ Thống Phân Tích & Phát Hiện Deepfake")

# ==========================================
# 2. LOAD MÔ HÌNH VÀ BỘ DETECTOR YUNET
# ==========================================
@st.cache_resource
def load_deepfake_model():
    model_file = 'best_deepfake_model.h5' if os.path.exists('best_deepfake_model.h5') else 'deepfake_model.h5'
    return tf.keras.models.load_model(model_file)

@st.cache_resource
def load_yunet_detector():
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, model_path)
    
    return cv2.FaceDetectorYN.create(
        model=model_path, config="", input_size=(320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )

try:
    model = load_deepfake_model()
    st.success("✅ Đã tải mô hình Deepfake MobileNetV2 thành công!")
except Exception as e:
    st.error(f"❌ Lỗi tải mô hình (.h5): {e}")
    st.stop()

yunet_detector = load_yunet_detector()

# ==========================================
# 3. SIDEBAR SELECTION
# ==========================================
st.sidebar.title("⚙️ Chế Độ Phân Tích")
option = st.sidebar.radio("Chọn loại dữ liệu đầu vào:", ["📷 Phân tích Ảnh", "🎥 Phân tích Video"])

# ==========================================
# 4. KHÂU CẮT ẢNH & TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
def extract_face_crop(img_rgb):
    """Trích xuất vùng mặt chuẩn tỉ lệ bằng YuNet"""
    h_img, w_img, _ = img_rgb.shape
    yunet_detector.setInputSize((w_img, h_img))
    
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, faces = yunet_detector.detect(img_bgr)

    if faces is not None and len(faces) > 0:
        best_face = max(faces, key=lambda f: f[-1])
        x, y, bw, bh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

        cx, cy = x + bw // 2, y + bh // 2
        crop_size = int(max(bw, bh) * 1.0)

        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w_img, cx + crop_size // 2)
        y2 = min(h_img, cy + crop_size // 2)

        cropped_rgb = img_rgb[y1:y2, x1:x2]
        if cropped_rgb.size > 0:
            return cropped_rgb

    return img_rgb

def predict_single_face(cropped_face_rgb, model):
    """Tiền xử lý và đưa qua mô hình MobileNetV2"""
    img_resized = cv2.resize(cropped_face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    input_tensor = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])
    return raw_pred

# ==========================================
# 5. CHỨC NĂNG 1: PHÂN TÍCH ẢNH ĐƠN
# ==========================================
if option == "📷 Phân tích Ảnh":
    st.subheader("📷 Phân tích Ảnh Đơn")
    uploaded_file = st.file_uploader("Tải ảnh cần kiểm tra...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Ảnh gốc tải lên", use_container_width=True)
            
            if st.button("🔍 Phân Tích Ảnh", use_container_width=True):
                with st.spinner("Đang trích xuất khuôn mặt bằng YuNet..."):
                    cropped_face = extract_face_crop(img_rgb)
                    st.image(cropped_face, caption="Khuôn mặt trích xuất", width=200)
                    
                    raw_p = predict_single_face(cropped_face, model)
                    
                    st.divider()
                    st.subheader("📊 Kết quả phân tích:")
                    st.write(f"• Điểm Sigmoid (Xác suất REAL): `{raw_p:.4f}`")
                    
                    if raw_p >= 0.50:
                        st.success("👉 KẾT LUẬN: **ẢNH THẬT (REAL)**")
                    else:
                        st.error("👉 KẾT LUẬN: **ẢNH GIẢ (DEEPFAKE)**")
                            
        except Exception as e:
            st.error(f"⚠️ Lỗi xử lý ảnh: {e}")

# ==========================================
# 6. CHỨC NĂNG 2: PHÂN TÍCH VIDEO
# ==========================================
else:
    st.subheader("🎥 Phân tích Video (Lọc Chuỗi Thời Gian Cải Tiến)")
    uploaded_video = st.file_uploader("Tải video cần kiểm tra...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button("🔍 Phân Tích Video", use_container_width=True):
            cap = cv2.VideoCapture(tfile.name)
            
            frame_count = 0
            analyzed_frames = 0
            raw_preds = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            FRAME_INTERVAL = 2  # Quét 1 frame mỗi 2 frame

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % FRAME_INTERVAL != 0:
                    continue
                
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress)
                status_text.text(f"Đang trích xuất & phân tích khung hình {frame_count}/{total_frames}...")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_face = extract_face_crop(frame_rgb)
                
                if cropped_face is not None:
                    raw_p = predict_single_face(cropped_face, model)
                    raw_preds.append(raw_p)
                    analyzed_frames += 1

            cap.release()
            try:
                os.remove(tfile.name)
            except Exception:
                pass
            
            status_text.text("Phân tích video hoàn thành!")
            st.divider()
            
            if analyzed_frames < 5:
                st.warning("⚠️ Không đủ khung hình phát hiện khuôn mặt để thực hiện đánh giá.")
            else:
                preds = np.array(raw_preds)
                median_val = float(np.median(preds))
                mean_val = float(np.mean(preds))

                # Đếm số frame nghi vấn Fake (< 0.45)
                fake_frames_mask = preds < 0.45
                fake_frame_count = int(np.sum(fake_frames_mask))
                fake_ratio = fake_frame_count / len(preds)

                # Tìm độ dài chuỗi frame Fake liên tiếp dài nhất
                max_consecutive_fake = 0
                current_run = 0
                for is_f in fake_frames_mask:
                    if is_f:
                        current_run += 1
                        if current_run > max_consecutive_fake:
                            max_consecutive_fake = current_run
                    else:
                        current_run = 0

                st.subheader("📊 Thông số thống kê video:")
                st.write(f"• Số frame phân tích: `{analyzed_frames}`")
                st.write(f"• Điểm Trung Vị (Median): `{median_val:.4f}`")
                st.write(f"• Điểm Trung Bình (Mean): `{mean_val:.4f}`")
                st.write(f"• Tỷ lệ frame nghi vấn Fake (< 0.45): `{fake_ratio * 100:.2f}%` ({fake_frame_count}/{analyzed_frames} frames)")
                st.write(f"• Chuỗi frame Fake liên tiếp dài nhất: `{max_consecutive_fake}` frames")

                # -------------------------------------------------------------
                # LOGIC ĐÁNH GIÁ CÂN BẰNG MỚI:
                # 
                # Video được coi là FAKE khi vi phạm MỘT trong các tiêu chuẩn:
                # 1. Median toàn bộ video < 0.50
                # 2. Có từ 5 frame LIÊN TIẾP trở lên bị rớt điểm (< 0.45)
                # 3. Tổng tỷ lệ frame rớt điểm chiếm từ 5% (0.05) trở lên
                # -------------------------------------------------------------
                has_consecutive_fake = (max_consecutive_fake >= 5)
                has_high_fake_ratio = (fake_ratio >= 0.05)
                is_median_fake = (median_val < 0.50)

                is_fake = is_median_fake or has_consecutive_fake or has_high_fake_ratio

                if is_fake:
                    st.error("👉 KẾT LUẬN: **VIDEO GIẢ (DEEPFAKE)**")
                else:
                    st.success("👉 KẾT LUẬN: **VIDEO THẬT (REAL)**")