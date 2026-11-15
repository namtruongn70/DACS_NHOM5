import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import streamlit as st
import tempfile

# ==========================================
# 0. CẤU HÌNH GIAO DIỆN & THIẾT BỊ
# ==========================================
st.set_page_config(
    page_title="Hệ thống Nhận diện Deepfake - ResNet50",
    page_icon="🛡️",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# Khởi tạo bộ dò khuôn mặt OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image_np):
    """
    Hàm cắt khuôn mặt thông minh: Cố gắng bắt mặt bằng Haar Cascade.
    Nếu không thấy (video góc nghiêng hoặc mờ), tự động lấy vùng trung tâm khung hình 
    (vì video deepfake thường tập trung khuôn mặt ở chính giữa).
    """
    H, W, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(W, x + w + padding_x)
        y2 = min(H, y + h + padding_y)
        
        cropped_face = image_np[y1:y2, x1:x2]
        if cropped_face.size > 0:
            return cropped_face
            
    # Dự phòng thông minh: Cắt vùng trung tâm (chiếm 60% khung hình) nếu không bắt được mặt bằng thuật toán
    margin_w = int(W * 0.2)
    margin_h = int(H * 0.2)
    cropped_fallback = image_np[margin_h:H-margin_h, margin_w:W-margin_w]
    if cropped_fallback.size > 0:
        return cropped_fallback
        
    return image_np

# ==========================================
# 1. ĐỊNH NGHĨA MÔ HÌNH VÀ TẢI TRỌNG SỐ AN TOÀN
# ==========================================
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
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_resnet50_model_torch.pth')
    
    if not os.path.exists(model_path):
        return None, model_path
    
    try:
        model = ResNet50Deepfake().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, model_path
    except Exception as e:
        st.error(f"Lỗi khi nạp trọng số mô hình: {str(e)}")
        return None, model_path

model, checked_path = load_model()

# Pipeline xử lý ảnh chuẩn ImageNet
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. GIAO DIỆN CHÍNH (UI)
# ==========================================
st.title("🛡️ HỆ THỐNG NHẬN DIỆN DEEPFAKE - PYTORCH RESNET50 (TỐI ƯU HÓA CAO)")
st.markdown("*(Đồ án tốt nghiệp - Nhận diện khuôn mặt giả mạo trên Ảnh và Video nâng cao)*")

if model is None:
    st.error(f"❌ Không tìm thấy file trọng số tại: `{checked_path}`")
    st.stop()
else:
    st.sidebar.success(f"✅ Mô hình sẵn sàng!\n- Thiết bị: {str(device).upper()}")

tab1, tab2 = st.tabs(["📷 Nhận diện Ảnh Tĩnh", "🎬 Nhận diện Video"])

# ==========================================
# TAB 1: NHẬN DIỆN ẢNH TĨNH
# ==========================================
with tab1:
    st.subheader("Tải lên hình ảnh cần kiểm tra")
    uploaded_image = st.file_uploader("Chọn file ảnh...", type=["jpg", "jpeg", "png", "webp"], key="img_up")

    if uploaded_image is not None:
        try:
            col1, col2 = st.columns(2)
            pil_original = Image.open(uploaded_image).convert('RGB')
            img_np = np.array(pil_original)
            
            face_np = extract_face(img_np)
            face_pil = Image.fromarray(face_np)
            
            with col1:
                st.image(pil_original, caption="Ảnh gốc tải lên", use_container_width=True)
                st.image(face_pil, caption="Vùng khuôn mặt trích xuất (AI Input)", width=200)
                
            with col2:
                with st.spinner("⏳ Đang phân tích..."):
                    img_tensor = transform(face_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(img_tensor).squeeze(1)
                        prob = torch.sigmoid(output).item()
                    
                    threshold = 0.53
                    is_real = prob >= threshold
                    label = "REAL (Ảnh Thật)" if is_real else "FAKE (Ảnh Giả Mạo - Deepfake)"
                    confidence = prob if is_real else (1 - prob)
                    
                    st.markdown("### 📊 Kết quả phân tích:")
                    if is_real:
                        st.success(f"**Nhãn dự đoán:** {label}")
                    else:
                        st.error(f"**Nhãn dự đoán:** {label}")
                    
                    st.metric(label="Độ tin cậy", value=f"{confidence * 100:.2f}%")
                    st.info(f"Xác suất gốc: {prob:.4f} (Ngưỡng: {threshold})")
        except Exception as ex:
            st.error(f"⚠️ Lỗi xử lý ảnh: {str(ex)}")

# ==========================================
# TAB 2: NHẬN DIỆN VIDEO (NÂNG CẤP THUẬT TOÁN QUÉT)
# ==========================================
with tab2:
    st.subheader("Tải lên video cần kiểm tra")
    uploaded_video = st.file_uploader("Chọn file video...", type=["mp4", "avi", "mov"], key="vid_up")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        
        if st.button("🚀 Bắt đầu phân tích Video Chuyên Sâu"):
            st.info("Đang quét toàn bộ khung hình video với độ nhạy cao...")
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            
            frame_count = 0
            fake_count = 0
            real_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Quét dày hơn (10 frame/giây) để không bỏ lỡ các phân đoạn deepfake ngắn
            skip_frames = max(1, int(fps / 10)) 
            preds_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    face_np = extract_face(image_rgb)
                    pil_img = Image.fromarray(face_np)
                    
                    img_tensor = transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_tensor).squeeze(1)
                        prob = torch.sigmoid(output).item()
                    
                    if prob >= 0.53:
                        real_count += 1
                    else:
                        fake_count += 1
                    preds_history.append(prob)
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_count / total_frames))
                status_text.text(f"Đang xử lý frame {frame_count}/{total_frames}...")
            
            cap.release()
            progress_bar.empty()
            status_text.empty()
            
            total_checked = real_count + fake_count
            if total_checked > 0:
                avg_prob = np.mean(preds_history)
                min_prob = np.min(preds_history) # Điểm xác suất thấp nhất trong video
                fake_ratio = (fake_count / total_checked) * 100
                
                # 💡 CHIẾN LƯỢC PHÁN QUYẾT NÂNG CAO CHO VIDEO DEEPFAKE TINH VI:
                # Video bị coi là FAKE nếu:
                # 1. Xác suất trung bình dưới ngưỡng 0.53, HOẶC
                # 2. Tỷ lệ frame bị nhận diện là giả chiếm trên 15%, HOẶC
                # 3. Có ít nhất một phân đoạn trong video có xác suất tụt xuống dưới 0.40 (dấu hiệu deepfake chớp nhoáng).
                is_video_fake = (avg_prob < 0.53) or (fake_ratio > 15.0) or (min_prob < 0.40)
                
                st.markdown("---")
                st.subheader("📋 Báo cáo tổng hợp Video:")
                
                col_v1, col_v2, col_v3 = st.columns(3)
                col_v1.metric("Tổng số Frame quét", total_checked)
                col_v2.metric("Số Frame là Fake", f"{fake_count} ({fake_ratio:.1f}%)")
                col_v3.metric("Số Frame là Real", f"{real_count} ({100 - fake_ratio:.1f}%)")
                
                st.info(f"📈 **Xác suất trung bình toàn video:** {avg_prob:.4f} | **Xác suất thấp nhất (Min):** {min_prob:.4f}")
                
                if is_video_fake:
                    st.error("🚨 **KẾT LUẬN: Video có chứa dấu hiệu Deepfake / Giả mạo!**")
                else:
                    st.success("✅ **KẾT LUẬN: Video hoàn toàn tự nhiên (Real)!**")
            else:
                st.warning("⚠️ Không đọc được khung hình nào từ video này.")