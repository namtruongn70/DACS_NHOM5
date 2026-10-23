import os
import json
import urllib.request
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from collections import deque
from keras.models import load_model
from streamlit_webrtc import (
    RTCConfiguration,
    VideoHTMLAttributes,
    WebRtcMode,
    webrtc_streamer,
)

# Tắt log rác TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------------------------------------------
# 1. CẤU HÌNH THAM SỐ TỐI ƯU VÀ CHỐNG BÁO SAI
# ---------------------------------------------------------
FRAME_WINDOW_SIZE = 10  # Bộ đệm 10 frame gần nhất
FAKE_THRESHOLD = 70.0   # Tăng lên 70% mới kết luận FAKE (chống báo sai mặt thật)

prediction_history = deque(maxlen=FRAME_WINDOW_SIZE)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------
# 2. TẢI MÔ HÌNH DỰ ĐOÁN & YUNET FACE DETECTOR
# ---------------------------------------------------------
@st.cache_resource
def load_deepfake_model_and_labels():
    model_file = 'best_deepfake_model.h5'
    if not os.path.exists(model_file):
        model_file = 'deepfake_model.h5'
    
    model = load_model(model_file)
    
    labels_file = 'class_indices.json'
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            class_indices = json.load(f)
    else:
        class_indices = {'fake': 0, 'real': 1}
        
    return model, class_indices

@st.cache_resource
def load_yunet_detector():
    """Tải YuNet Face Detector - Bắt góc nghiêng tốt"""
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, model_path)
    
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector

try:
    model, class_indices = load_deepfake_model_and_labels()
    # Xác định rõ index của lớp 'fake'
    fake_class_idx = class_indices.get('fake', 0)
except Exception as e:
    model = None
    fake_class_idx = 0

yunet_detector = load_yunet_detector()

# ---------------------------------------------------------
# 3. TRÍCH XUẤT MẶT BẰNG YUNET
# ---------------------------------------------------------
def detect_and_crop_face_yunet(img_bgr):
    """Trích xuất khuôn mặt bằng YuNet"""
    h, w, _ = img_bgr.shape
    yunet_detector.setInputSize((w, h))
    _, faces = yunet_detector.detect(img_bgr)

    if faces is not None and len(faces) > 0:
        face = max(faces, key=lambda f: f[-1])
        x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

        # Padding 20% giữ tỉ lệ chuẩn
        padding_w = int(0.20 * bw)
        padding_h = int(0.20 * bh)

        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(w, x + bw + padding_w)
        y2 = min(h, y + bh + padding_h)

        cropped_bgr = img_bgr[y1:y2, x1:x2]
        if cropped_bgr.size > 0:
            return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB), (x, y, bw, bh)

    return None, None

def process_video_frame(frame):
    img_bgr = frame.to_ndarray(format="bgr24")

    if model is not None:
        try:
            cropped_face_rgb, face_box = detect_and_crop_face_yunet(img_bgr)
            
            if cropped_face_rgb is not None:
                # Preprocess & Model Predict
                img_resized = cv2.resize(cropped_face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                img_array = img_resized.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(img_array, axis=0)

                raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])

                # Tính chính xác xác suất FAKE (0.0 -> 100.0%)
                if fake_class_idx == 1:
                    s_fake_frame = raw_pred * 100.0
                else:
                    s_fake_frame = (1.0 - raw_pred) * 100.0

                prediction_history.append(s_fake_frame)
                
                # Tính trung bình động trong 10 frame gần nhất
                avg_fake_score = sum(prediction_history) / len(prediction_history)
                avg_real_score = 100.0 - avg_fake_score

                # Đánh giá dựa trên FAKE_THRESHOLD (70%)
                if avg_fake_score >= FAKE_THRESHOLD:
                    label = f"FAKE ({avg_fake_score:.1f}%)"
                    color = (0, 0, 255) # Đỏ
                else:
                    label = f"REAL ({avg_real_score:.1f}%)"
                    color = (0, 255, 0) # Xanh lá

                # Vẽ khung ôm mặt
                x, y, bw, bh = face_box
                cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), color, 2)

                # Hiển thị Banner kết quả
                cv2.rectangle(img_bgr, (10, 10), (380, 65), (0, 0, 0), -1)
                cv2.putText(img_bgr, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img_bgr, (10, 10), (380, 65), (0, 0, 0), -1)
                cv2.putText(img_bgr, "SEARCHING FACE...", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        except Exception as e:
            cv2.putText(img_bgr, "AI Processing Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame.from_ndarray(img_bgr, format="bgr24")

# ---------------------------------------------------------
# 4. GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Deepfake Detection Video Call", layout="wide")
st.title("📹 Hệ Thống Video Call Giám Sát Deepfake AI")

if model:
    st.success("✅ Đã kết nối thành công Mô hình AI & YuNet Detector!")

room_id = st.text_input("Nhập Mã Phòng Video Call:", "demo123")
st.write(f"🔑 Mã phòng hiện tại: **{room_id}**")

st.markdown("---")

role = st.radio(
    "Chọn vai trò cho Tab này:",
    ("Người phát (Sender / Kẻ tấn công)", "Hệ thống giám sát (Receiver / AI Detector)"),
    horizontal=True
)

st.markdown("---")

if role == "Người phát (Sender / Kẻ tấn công)":
    st.subheader("📤 Luồng phát Video (Sender)")
    st.info("📌 **Hướng dẫn:** Chọn camera (hoặc phần mềm Deepfake/OBS Virtual Cam) và bấm **START**.")

    webrtc_streamer(
        key=f"sender-{room_id}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=True),
    )
else:
    st.subheader("📥 Luồng xem & Phân tích AI (Receiver)")
    st.info("📌 **Hướng dẫn:** Bấm **START** để nhận luồng video và AI sẽ tiến hành phân tích trực tiếp.")

    webrtc_streamer(
        key=f"receiver-{room_id}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=process_video_frame,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=False),
    )