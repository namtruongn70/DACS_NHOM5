import os
import time
import threading
import warnings
from collections import deque

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from streamlit_webrtc import (
    RTCConfiguration,
    VideoHTMLAttributes,
    WebRtcMode,
    webrtc_streamer,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CẤU HÌNH NÂNG NGƯỠNG ĐỂ GIẢM LỌT FAKE TRÊN WEBRTC
# ---------------------------------------------------------
MODEL_PATH = "best_convnext_tiny_deepfake.pth"

IMG_SIZE = 224
FRAME_WINDOW_SIZE = 5        # Cửa sổ 5 frame gần nhất
REAL_THRESHOLD = 0.75        # Nâng cao ngưỡng REAL: prob_real >= 0.75 mới là REAL; < 0.75 nghi/báo FAKE
SUSPECT_REAL_TH = 0.85       # Frame có prob_real < 0.85 được tính là frame nghi vấn FAKE
MIN_SUSPECT_HITS = 2         # Có >= 2 frame nghi vấn trong cửa sổ -> báo FAKE
INVERT_PROB = False          # Giữ False mặc định để không đảo ngược đồng loạt khi RawReal cao
DETECT_SCALE = 0.4           # Thu nhỏ ảnh gray khi dò mặt để giữ FPS ổn định
INFERENCE_STRIDE = 1         # 1 = chạy mỗi frame có mặt
USE_FP16_CUDA = False        # Tắt mặc định để giữ độ chính xác số học ổn định
BLUR_VAR_THRESHOLD = 6.0

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
STD_TENSOR = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

state_lock = threading.Lock()
runtime_state = {
    "real_history": deque(maxlen=FRAME_WINDOW_SIZE),
    "latency": deque(maxlen=30),
    "last_frame_time": None,
    "frame_counter": 0,
    "last_render_state": None,
    "last_real_avg": None,
    "last_real_min": None,
    "real_threshold": REAL_THRESHOLD,
    "suspect_real_th": SUSPECT_REAL_TH,
    "min_suspect_hits": MIN_SUSPECT_HITS,
    "invert_prob": INVERT_PROB,
    "stride": INFERENCE_STRIDE,
    "use_fp16": USE_FP16_CUDA,
}


# ---------------------------------------------------------
# 2. MÔ HÌNH CONVNEXT-TINY & DÒ MẶT
# ---------------------------------------------------------
class ConvNeXtTinyDeepfake(nn.Module):
    def __init__(self, dropout_p=0.35):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)


@st.cache_resource
def load_convnext_model():
    if not os.path.exists(MODEL_PATH):
        return None
    m = ConvNeXtTinyDeepfake(dropout_p=0.35).to(device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


model = load_convnext_model()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(image_rgb):
    H, W, _ = image_rgb.shape
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    scale = DETECT_SCALE if 0.25 <= DETECT_SCALE < 1.0 else 1.0
    if scale < 1.0:
        small_w = max(160, int(W * scale))
        small_h = max(120, int(H * scale))
        gray_detect = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        inv_x = W / float(small_w)
        inv_y = H / float(small_h)
    else:
        gray_detect = gray
        inv_x = 1.0
        inv_y = 1.0

    faces = face_cascade.detectMultiScale(
        gray_detect, scaleFactor=1.15, minNeighbors=3, minSize=(25, 25)
    )

    if len(faces) > 0:
        (sx, sy, sw, sh) = max(faces, key=lambda f: f[2] * f[3])
        x = int(sx * inv_x)
        y = int(sy * inv_y)
        w = int(sw * inv_x)
        h = int(sh * inv_y)

        pad_x = int(w * 0.14)
        pad_y = int(h * 0.14)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)

        cropped_face = image_rgb[y1:y2, x1:x2]
        if cropped_face.size > 0:
            face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            if laplacian_var < BLUR_VAR_THRESHOLD:
                return cropped_face, (x1, y1, x2, y2), False, True
            return cropped_face, (x1, y1, x2, y2), True, False

    return image_rgb, (0, 0, W, H), False, False


def preprocess_face_tensor(face_rgb):
    resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor - MEAN_TENSOR) / STD_TENSOR
    return tensor


# ---------------------------------------------------------
# 3. XỬ LÝ KHUNG HÌNH WEBRTC
# ---------------------------------------------------------
def process_video_frame(frame):
    t_start = time.perf_counter()

    with state_lock:
        runtime_state["frame_counter"] += 1
        frame_counter = runtime_state["frame_counter"]
        now = time.perf_counter()
        prev_time = runtime_state["last_frame_time"]
        inst_fps = (1.0 / (now - prev_time)) if (prev_time is not None and now > prev_time) else 0.0
        runtime_state["last_frame_time"] = now

        real_th = runtime_state["real_threshold"]
        suspect_th = runtime_state["suspect_real_th"]
        min_hits = runtime_state["min_suspect_hits"]
        invert_prob = runtime_state["invert_prob"]
        stride = max(1, int(runtime_state["stride"]))
        use_fp16 = runtime_state["use_fp16"]
        last_render_state = runtime_state["last_render_state"]

    img_bgr = frame.to_ndarray(format="bgr24")

    if model is not None:
        try:
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_np, (x1, y1, x2, y2), face_found, is_blurred = extract_face(image_rgb)

            if not face_found or is_blurred:
                with state_lock:
                    runtime_state["real_history"].clear()
                    runtime_state["last_render_state"] = None
                    runtime_state["last_real_avg"] = None
                    runtime_state["last_real_min"] = None

                warning_text = "WARNING: MOVING TOO FAST" if is_blurred else "NO FACE DETECTED"
                cv2.rectangle(img_bgr, (10, 10), (450, 65), (0, 140, 255), -1)
                cv2.putText(
                    img_bgr, warning_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )
            else:
                should_infer = (
                    (stride == 1)
                    or (frame_counter % stride == 0)
                    or (last_render_state is None)
                )

                if should_infer:
                    img_tensor = preprocess_face_tensor(face_np)

                    with torch.no_grad():
                        if use_fp16 and device.type == "cuda":
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                output = model(img_tensor).squeeze(1)
                        else:
                            output = model(img_tensor).squeeze(1)
                        raw_prob_real = float(torch.sigmoid(output.float()).item())

                    prob_real = (1.0 - raw_prob_real) if invert_prob else raw_prob_real

                    with state_lock:
                        runtime_state["real_history"].append(prob_real)
                        hist_arr = np.array(list(runtime_state["real_history"]), dtype=np.float32)
                        real_avg = float(np.mean(hist_arr))
                        real_min = float(np.min(hist_arr))
                        suspect_hits = int(np.sum(hist_arr < suspect_th))
                        runtime_state["last_real_avg"] = real_avg
                        runtime_state["last_real_min"] = real_min

                    # Chỉ coi là REAL khi trung bình prob_real >= real_th VÀ số frame rớt dưới suspect_th ít hơn min_hits
                    is_real = (real_avg >= real_th) and (suspect_hits < min_hits)

                    if not is_real:
                        # Mức độ nghi ngờ FAKE tỷ lệ với độ hụt so với real_th (hoặc dùng real_min)
                        fake_score = max(0.0, min(1.0, 1.0 - min(real_avg, real_min)))
                        label = f"FAKE - DEEPFAKE ({fake_score * 100:.1f}%)"
                        color = (0, 0, 255)
                    else:
                        real_conf = max(0.0, min(1.0, real_avg))
                        label = f"REAL ({real_conf * 100:.1f}%)"
                        color = (0, 255, 0)

                    with state_lock:
                        runtime_state["last_render_state"] = (label, color)
                else:
                    with state_lock:
                        label, color = runtime_state["last_render_state"]

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img_bgr, (10, 10), (470, 65), (0, 0, 0), -1)
                cv2.putText(
                    img_bgr, label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA
                )

        except Exception as e:
            print(f"❌ Lỗi xử lý frame: {e}")
            cv2.putText(
                img_bgr, "AI Processing Error", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    with state_lock:
        runtime_state["latency"].append(elapsed_ms)
        lat_list = list(runtime_state["latency"])
        r_avg = runtime_state["last_real_avg"]
        r_min = runtime_state["last_real_min"]
        r_th = runtime_state["real_threshold"]
        s_th = runtime_state["suspect_real_th"]

    avg_lat = sum(lat_list) / len(lat_list)
    proc_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    r_avg_str = f"{r_avg:.3f}" if r_avg is not None else "N/A"
    r_min_str = f"{r_min:.3f}" if r_min is not None else "N/A"
    metric_text = (
        f"FPS:{proc_fps:.1f}/{inst_fps:.1f} | Lat:{avg_lat:.0f}ms | "
        f"RealAvg:{r_avg_str} | RealMin:{r_min_str} | RealTH:{r_th:.2f} | SusTH:{s_th:.2f}"
    )
    cv2.rectangle(img_bgr, (10, 72), (820, 105), (30, 30, 30), -1)
    cv2.putText(
        img_bgr, metric_text, (16, 94),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA
    )

    return frame.from_ndarray(img_bgr, format="bgr24")


# ---------------------------------------------------------
# 4. GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Giám sát Deepfake Video Call - ConvNeXt-Tiny", layout="wide")
st.title("Hệ Thống Video Call Giám Sát Deepfake AI (ConvNeXt-Tiny - Ngưỡng Cao)")

if model is not None:
    st.success(f"Đã kết nối mô hình ConvNeXt-Tiny ({device.type.upper()}) thành công!")
else:
    st.error(f"Không tìm thấy file trọng số `{MODEL_PATH}`!")

with st.expander("Tinh chỉnh ngưỡng nâng cao trên luồng thực tế"):
    c1, c2, c3 = st.columns(3)
    with c1:
        r_th_ui = st.slider("Ngưỡng REAL cao (RealAvg >= mức này mới là REAL)", 0.50, 0.99, REAL_THRESHOLD, 0.01)
        s_th_ui = st.slider("Ngưỡng frame nghi vấn (prob_real < mức này)", 0.50, 0.99, SUSPECT_REAL_TH, 0.01)
    with c2:
        hits_ui = st.slider("Số frame nghi vấn trong cửa sổ để báo FAKE", 1, 5, MIN_SUSPECT_HITS, 1)
        win_ui = st.slider("Kích thước cửa sổ frame", 1, 10, FRAME_WINDOW_SIZE, 1)
    with c3:
        stride_ui = st.slider("Inference Stride", 1, 4, INFERENCE_STRIDE, 1)
        inv_ui = st.checkbox("Đảo chiều xác suất (INVERT_PROB)", value=INVERT_PROB)
        fp16_ui = st.checkbox("Dùng FP16 trên CUDA", value=USE_FP16_CUDA)

    with state_lock:
        runtime_state["real_threshold"] = r_th_ui
        runtime_state["suspect_real_th"] = s_th_ui
        runtime_state["min_suspect_hits"] = hits_ui
        runtime_state["invert_prob"] = inv_ui
        runtime_state["stride"] = stride_ui
        runtime_state["use_fp16"] = fp16_ui
        if runtime_state["real_history"].maxlen != win_ui:
            runtime_state["real_history"] = deque(list(runtime_state["real_history"]), maxlen=win_ui)

room_id = st.text_input("Nhập Mã Phòng Video Call:", "room_convnext_high_th")
st.write(f"Mã phòng hiện tại: **{room_id}**")

st.markdown("---")

role = st.radio(
    "Chọn vai trò cho Tab này:",
    ("Người phát (Sender / Kẻ tấn công)", "Hệ thống giám sát (Receiver / AI Detector)"),
    horizontal=True,
)

st.markdown("---")

video_constraints = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
    "audio": False,
}

if role == "Người phát (Sender / Kẻ tấn công)":
    st.subheader("Luồng phát Video (Sender)")
    st.info("Chọn camera và bấm **START**.")

    webrtc_streamer(
        key=f"sender-highth-{room_id}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=video_constraints,
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=True),
    )
else:
    st.subheader("Luồng xem & Phân tích AI (Receiver)")
    st.info("Bấm **START** để nhận luồng video và phân tích real-time.")

    webrtc_streamer(
        key=f"receiver-highth-{room_id}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=process_video_frame,
        media_stream_constraints=video_constraints,
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=False),
    )