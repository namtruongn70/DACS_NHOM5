import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

# ==========================================
# 0. CẤU HÌNH THIẾT BỊ & SEED
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Tối ưu cho kích thước ảnh cố định 224x224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
print(f"🚀 Thiết bị: {device} | AMP (fp16): {use_amp}")
if device.type == "cuda":
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. ĐƯỜNG DẪN DỮ LIỆU CŨ & MỚI
# ==========================================
MAIN_TRAIN_DIR = r"F:\archive\real_vs_fake\real-vs-fake\train"
MAIN_VAL_DIR   = r"F:\archive\real_vs_fake\real-vs-fake\valid"

EXTRA_FAKE_DIR = r"F:\archivebx\real_and_fake_face\training_fake"
EXTRA_REAL_DIR = r"F:\archivebx\real_and_fake_face\training_real"

# Giữ nguyên toàn bộ dữ liệu để không giảm hiệu năng
MAX_MAIN_TRAIN_PER_CLASS = 50000
MAX_MAIN_VAL_PER_CLASS   = 10000
EXTRA_VAL_RATIO = 0.20

IMG_SIZE = 224
BATCH_SIZE = 16       # Giảm từ 32 -> 16 để nhẹ VRAM cho MX350
ACCUM_STEPS = 2       # Tích lũy 2 bước => Effective Batch Size vẫn là 32
EPOCHS_WARMUP = 2
EPOCHS_FINE = 8
LR_WARMUP = 5e-4
LR_FINE = 8e-5

SAVE_MODEL_PATH = "best_convnext_tiny_deepfake.pth"
SAVE_META_PATH  = "convnext_tiny_meta.json"

USER_PREFIXES = ("user_fake", "fake_user", "user", "real_user", "user_real")
VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Quy ước nhãn thống nhất: 0 = fake, 1 = real
CLASS_TO_IDX = {"fake": 0, "real": 1}

# ==========================================
# 2. NẠP DỮ LIỆU ĐA NGUỒN (BẢO TOÀN 100% USER_*)
# ==========================================
def build_dataset_lists(is_train=True):
    filepaths = []
    labels = []
    rng = random.Random(SEED)

    main_dir = MAIN_TRAIN_DIR if is_train else MAIN_VAL_DIR
    max_main = MAX_MAIN_TRAIN_PER_CLASS if is_train else MAX_MAIN_VAL_PER_CLASS

    extra_dirs = {
        "fake": EXTRA_FAKE_DIR,
        "real": EXTRA_REAL_DIR,
    }

    for cls_name, cls_label in CLASS_TO_IDX.items():
        main_user_files = []
        main_other_files = []
        cls_main_dir = os.path.join(main_dir, cls_name)

        if os.path.isdir(cls_main_dir):
            all_files = sorted([
                f for f in os.listdir(cls_main_dir)
                if f.lower().endswith(VALID_EXTS)
            ])
            main_user_files = [
                f for f in all_files
                if any(prefix in f.lower() for prefix in USER_PREFIXES)
            ]
            main_other_files = [f for f in all_files if f not in main_user_files]
            rng.shuffle(main_other_files)

            needed_other = max(0, max_main - len(main_user_files))
            chosen_main = main_user_files + main_other_files[:needed_other]

            for f in chosen_main:
                filepaths.append(os.path.join(cls_main_dir, f))
                labels.append(cls_label)
        else:
            chosen_main = []

        chosen_extra = []
        extra_dir = extra_dirs.get(cls_name)
        if extra_dir and os.path.isdir(extra_dir):
            extra_all = sorted([
                f for f in os.listdir(extra_dir)
                if f.lower().endswith(VALID_EXTS)
            ])
            rng.shuffle(extra_all)
            split_idx = int(len(extra_all) * (1.0 - EXTRA_VAL_RATIO))
            chosen_extra = extra_all[:split_idx] if is_train else extra_all[split_idx:]

            for f in chosen_extra:
                filepaths.append(os.path.join(extra_dir, f))
                labels.append(cls_label)

        split_tag = "TRAIN" if is_train else "VAL"
        print(
            f"[{split_tag} | {cls_name.upper()}] "
            f"User: {len(main_user_files)} | "
            f"140k: {len(chosen_main) - len(main_user_files)} | "
            f"Extra: {len(chosen_extra)} => Tổng: {len(chosen_main) + len(chosen_extra)}"
        )

    return filepaths, labels

print("📌 Đang tổng hợp danh sách dữ liệu...")
train_paths, train_labels = build_dataset_lists(is_train=True)
val_paths, val_labels     = build_dataset_lists(is_train=False)

with open("class_indices_convnext.json", "w", encoding="utf-8") as f:
    json.dump({"fake": 0, "real": 1}, f, indent=2)

# ==========================================
# 3. DATASET & AUGMENTATION SÁT THỰC TẾ
# ==========================================
class DeepfakeFaceDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=8),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = DeepfakeFaceDataset(train_paths, train_labels, transform=train_transform)
val_dataset   = DeepfakeFaceDataset(val_paths, val_labels, transform=val_transform)

num_workers = 0 if os.name == "nt" else 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

# ==========================================
# 4. MÔ HÌNH CONVNEXT-TINY
# ==========================================
class ConvNeXtTinyDeepfake(nn.Module):
    def __init__(self, dropout_p=0.35):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)

def set_backbone_trainable(model, trainable=True):
    for name, param in model.backbone.named_parameters():
        if not name.startswith("classifier.2"):
            param.requires_grad = trainable

model = ConvNeXtTinyDeepfake(dropout_p=0.35).to(device)
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

def run_epoch(loader, optimizer=None, is_train=True, accum_steps=1):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "Train" if is_train else "Val"
    loop = tqdm(loader, desc=desc, leave=False)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with ctx:
        for step, (images, labels) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / accum_steps if is_train else raw_loss

            if is_train:
                scaler.scale(loss).backward()
                is_accum_boundary = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
                if is_accum_boundary:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            total_loss += raw_loss.item() * batch_size
            probs = torch.sigmoid(logits.float())
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(loss=f"{raw_loss.item():.4f}", acc=f"{correct / max(1, total):.4f}")

    return total_loss / max(1, total), correct / max(1, total)

# ==========================================
# 5. GIAI ĐOẠN 1: WARM-UP HEAD
# ==========================================
print("\n🚀 --- GIAI ĐOẠN 1: WARM-UP CLASSIFIER HEAD ---")
set_backbone_trainable(model, trainable=False)
optimizer_warmup = optim.AdamW(model.backbone.classifier[2].parameters(), lr=LR_WARMUP, weight_decay=1e-4)

for epoch in range(1, EPOCHS_WARMUP + 1):
    tr_loss, tr_acc = run_epoch(train_loader, optimizer=optimizer_warmup, is_train=True, accum_steps=ACCUM_STEPS)
    va_loss, va_acc = run_epoch(val_loader, optimizer=None, is_train=False, accum_steps=1)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    print(
        f"Warm-up [{epoch}/{EPOCHS_WARMUP}] "
        f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
        f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f}"
    )

# ==========================================
# 6. GIAI ĐOẠN 2: FINE-TUNING TOÀN BỘ MẠNG
# ==========================================
print("\n🔓 --- GIAI ĐOẠN 2: FINE-TUNING TOÀN BỘ CONVNEXT-TINY ---")
set_backbone_trainable(model, trainable=True)
optimizer_fine = optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_fine, T_max=EPOCHS_FINE, eta_min=1e-6)

best_val_acc = 0.0
patience = 3
no_improve = 0

for epoch in range(1, EPOCHS_FINE + 1):
    tr_loss, tr_acc = run_epoch(train_loader, optimizer=optimizer_fine, is_train=True, accum_steps=ACCUM_STEPS)
    va_loss, va_acc = run_epoch(val_loader, optimizer=None, is_train=False, accum_steps=1)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    current_lr = optimizer_fine.param_groups[0]["lr"]
    print(
        f"Fine-tune [{epoch}/{EPOCHS_FINE}] (lr={current_lr:.2e}) "
        f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
        f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f}"
    )

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        no_improve = 0
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"💾 Lưu model tốt nhất tại: {SAVE_MODEL_PATH} (Val Acc: {best_val_acc * 100:.2f}%)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⏹️ Dừng sớm do Val Accuracy không cải thiện thêm.")
            break

# ==========================================
# 7. ĐÁNH GIÁ & TÌM NGƯỠNG TỐI ƯU
# ==========================================
print("\n📊 --- ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN VALIDATION ---")
try:
    state_dict = torch.load(SAVE_MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(SAVE_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

all_probs = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.numpy())

all_probs = np.array(all_probs)
all_targets = np.array(all_targets).astype(int)

best_th = 0.5
best_score = -1.0
for th in np.linspace(0.20, 0.80, 61):
    preds_th = (all_probs >= th).astype(int)
    score = f1_score(all_targets, preds_th, average="macro")
    if score > best_score:
        best_score = score
        best_th = float(th)

final_preds = (all_probs >= best_th).astype(int)
final_acc = accuracy_score(all_targets, final_preds)

print(f"✅ Val Accuracy (tại threshold={best_th:.2f}): {final_acc * 100:.2f}% | Macro F1: {best_score:.4f}")
print("-" * 65)
print(classification_report(all_targets, final_preds, target_names=["fake", "real"], digits=4))

meta_info = {
    "model_arch": "convnext_tiny",
    "img_size": IMG_SIZE,
    "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
    "label_mapping": {"fake": 0, "real": 1},
    "best_threshold": round(best_th, 4),
    "best_val_acc_default_05": round(float(best_val_acc), 6),
    "val_acc_at_best_th": round(float(final_acc), 6),
}
with open(SAVE_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta_info, f, indent=2)
print(f"📝 Đã lưu cấu hình ngưỡng & metadata vào: {SAVE_META_PATH}")

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train Acc", marker="o")
plt.plot(history["val_acc"], label="Val Acc", marker="s")
plt.title(f"ConvNeXt-Tiny Accuracy (Best Val: {best_val_acc * 100:.2f}%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train Loss", marker="o")
plt.plot(history["val_loss"], label="Val Loss", marker="s")
plt.title("ConvNeXt-Tiny Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("training_history_convnext.png")
print("📈 Đã lưu biểu đồ vào: training_history_convnext.png")