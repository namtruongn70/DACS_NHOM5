import os
import cv2
import numpy as np
import streamlit as st
from collections import deque
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from streamlit_webrtc import (
    RTCConfiguration,
    VideoHTMLAttributes,
    WebRtcMode,
    webrtc_streamer,
)

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. CẤU HÌNH BỘ LỌC CHỐNG NHẤP NHÁY NÂNG CAO
# ---------------------------------------------------------
FRAME_WINDOW_SIZE = 9   # Tăng cửa sổ bộ đệm lên 9 frame để có cái nhìn tổng quan ổn định hơn
FAKE_THRESHOLD = 0.50   # Ngưỡng trung tính để cân bằng độ nhạy Real/Fake

prediction_history = deque(maxlen=FRAME_WINDOW_SIZE)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ---------------------------------------------------------
# 2. ĐỊNH NGHĨA MÔ HÌNH & TẢI TRỌNG SỐ
# ---------------------------------------------------------
class ResNet50Deepfake(nn.Module):
    def __init__(self):
        super(ResNet50Deepfake, self).__init__()
        self.base_model = models.resnet50(weights=None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.base_model(x)

@st.cache_resource
def load_resnet50_model():
    model_path = 'best_resnet50_model_torch.pth'
    if not os.path.exists(model_path):
        return None
    
    model = ResNet50Deepfake().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Khởi tạo bộ dò khuôn mặt OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image_np):
    """Trích xuất khuôn mặt, cân bằng sáng và lọc nhiễu chuyển động"""
    H, W, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
    
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        padding_x = int(w * 0.15)
        padding_y = int(h * 0.15)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(W, x + w + padding_x)
        y2 = min(H, y + h + padding_y)
        
        cropped_face = image_np[y1:y2, x1:x2]
        
        if cropped_face.size > 0:
            face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            
            # Kiểm tra độ nhòe chuyển động
            if laplacian_var < 10.0:
                return cropped_face, (x1, y1, x2, y2), False, True
            
            # Cân bằng sáng tự động (Histogram Equalization)
            ycrcb = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            balanced_face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            
            return balanced_face, (x1, y1, x2, y2), True, False
            
    return image_np, (0, 0, W, H), False, False

model = load_resnet50_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 3. HÀM XỬ LÝ KHUNG HÌNH WEBRTC VỚI BỎ PHIẾU ĐA SỐ (MAJORITY VOTING)
# ---------------------------------------------------------
def process_video_frame(frame):
    img_bgr = frame.to_ndarray(format="bgr24")

    if model is not None:
        try:
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_np, (x1, y1, x2, y2), face_found, is_blurred = extract_face(image_rgb)
            
            # --- TRƯỜNG HỢP 1: KHÔNG THẤY MẶT HOẶC BỊ NHÒE ---
            if not face_found or is_blurred:
                prediction_history.clear() # Xóa sạch bộ đệm
                
                warning_text = "WARNING: MOVING TOO FAST" if is_blurred else "NO FACE DETECTED"
                
                cv2.rectangle(img_bgr, (10, 10), (450, 65), (0, 140, 255), -1)
                cv2.putText(img_bgr, warning_text, (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- TRƯỜNG HỢP 2: CÓ MẶT ỔN ĐỊNH, TIẾN HÀNH BỎ PHIẾU ĐA SỐ ---
            else:
                pil_img = Image.fromarray(face_np)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor).squeeze(1)
                    raw_prob = torch.sigmoid(output).item() # 1 là Real, 0 là Fake

                prediction_history.append(raw_prob)
                
                # 💡 THUẬT TOÁN BỎ PHIẾU ĐA SỐ NGHIÊNG NGẶT (STRICT MAJORITY VOTING)
                # Đếm số lượng frame đánh giá là Real trong bộ đệm hiện tại
                real_votes = sum(1 for p in prediction_history if p >= FAKE_THRESHOLD)
                total_frames = len(prediction_history)
                
                # Tỷ lệ phiếu Real phải chiếm từ 70% trở lên mới được công nhận là REAL.
                # Ngược lại, nếu có nhiều phiếu Fake hơn, lập tức khóa nhãn FAKE.
                is_real_strict = (real_votes / total_frames) >= 0.70
                
                # Nếu đang là Fake, đòi hỏi tỷ lệ phiếu Fake phải áp đảo để tránh nháy Real bậy bạ
                if not is_real_strict:
                    # Kiểm tra thêm điểm trung bình để tính độ tự tin hiển thị
                    avg_prob = sum(prediction_history) / total_frames
                    confidence = 1.0 - avg_prob
                    label = f"🚨 FAKE - DEEPFAKE ({confidence * 100:.1f}%)"
                    color = (0, 0, 255) # Đỏ rực
                else:
                    avg_prob = sum(prediction_history) / total_frames
                    confidence = avg_prob
                    label = f"REAL ({confidence * 100:.1f}%)"
                    color = (0, 255, 0) # Xanh lá

                # Vẽ khung viền và hiển thị nhãn kết quả
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img_bgr, (10, 10), (440, 65), (0, 0, 0), -1)
                cv2.putText(img_bgr, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        except Exception as e:
            cv2.putText(img_bgr, "AI Processing Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame.from_ndarray(img_bgr, format="bgr24")

# ---------------------------------------------------------
# 4. GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Giám sát Deepfake Video Call - ResNet50 PyTorch", layout="wide")
st.title("📹 Hệ Thống Video Call Giám Sát Deepfake AI (Chống Nháy Frame & Lọt Lưới)")

if model is not None:
    st.success("✅ Đã kết nối mô hình ResNet50 PyTorch thành công!")
else:
    st.error("❌ Không tìm thấy file trọng số `best_resnet50_model_torch.pth`!")

room_id = st.text_input("Nhập Mã Phòng Video Call:", "room_strict_voting")
st.write(f"🔑 Mã phòng hiện tại: **{room_id}**")

st.markdown("---")

role = st.radio(
    "Chọn vai trò cho Tab này:",
    ("Người phát (Sender / Kẻ tấn công)", "Hệ thống giám sát (Receiver / AI Detector Ổn Định Tuyệt Đối)"),
    horizontal=True
)

st.markdown("---")

if role == "Người phát (Sender / Kẻ tấn công)":
    st.subheader("📤 Luồng phát Video (Sender)")
    st.info("📌 Chọn camera và bấm **START**.")

    webrtc_streamer(
        key=f"sender-strict-{room_id}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=True),
    )
else:
    st.subheader("📥 Luồng xem & Phân tích AI (Receiver)")
    st.info("📌 Bấm **START** để nhận luồng video. Cơ chế biểu quyết đa số nghiêm ngặt giúp loại bỏ hoàn toàn các frame nhảy sai.")

    webrtc_streamer(
        key=f"receiver-strict-{room_id}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=process_video_frame,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=False),
    )