import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
from scipy.ndimage import binary_erosion

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
LOG_DIR = ROOT_DIR / "results" / "logs" 
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(ROOT_DIR)

def set_seed(seed=2028):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2028)

def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# CORREÇÃO 1: Erosão adaptativa proporcional ao tamanho de cada instância
def build_three_class_target(mask_paths, h, w):
    target = np.zeros((h, w), dtype=np.uint8)
    all_interiors = np.zeros((h, w), dtype=bool)
    all_boundaries = np.zeros((h, w), dtype=bool)

    struct = np.ones((3, 3), dtype=bool)

    for m_path in mask_paths:
        mask = np.array(Image.open(m_path)) > 0
        area = np.sum(mask)

        if area == 0:
            continue

        # Calcula o raio equivalente para determinar o número seguro de iterações
        equiv_radius = np.sqrt(area / np.pi)
        erosion_iters = int(np.clip(np.floor(equiv_radius * 0.25), 1, 3))

        interior = binary_erosion(mask, structure=struct, iterations=erosion_iters)
        boundary = mask & ~interior

        all_interiors |= interior
        all_boundaries |= boundary

    # 1 = interior, 2 = boundary (prioridade para sobreposição)
    target[all_interiors] = 1
    target[all_boundaries] = 2

    return target

class DSB2018Dataset(Dataset):
    def __init__(self, raw_dir, sample_ids, transform=None):
        self.raw_dir = Path(raw_dir)
        self.sample_ids = sample_ids
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_folder = self.raw_dir / self.sample_ids[idx]

        img_path = list((sample_folder / "images").glob("*.png"))[0]
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_paths = list((sample_folder / "masks").glob("*.png"))
        h, w = image.shape[:2]

        target = build_three_class_target(mask_paths, h, w)

        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"].long()

        return image, target, self.sample_ids[idx]

# CORREÇÃO 2: Acumulador global de métricas (evita viés de média de batch)
class MetricTracker:
    def __init__(self, num_classes=3, smooth=1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.total_intersections = np.zeros(self.num_classes, dtype=np.float64)
        self.total_unions = np.zeros(self.num_classes, dtype=np.float64)
        self.total_pred_areas = np.zeros(self.num_classes, dtype=np.float64)
        self.total_target_areas = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)

            intersection = (pred_c & target_c).sum().item()
            pred_area = pred_c.sum().item()
            target_area = target_c.sum().item()
            union = pred_area + target_area - intersection

            self.total_intersections[c] += intersection
            self.total_unions[c] += union
            self.total_pred_areas[c] += pred_area
            self.total_target_areas[c] += target_area

    def compute(self):
        ious = (self.total_intersections + self.smooth) / (self.total_unions + self.smooth)
        dices = (2 * self.total_intersections + self.smooth) / (
            self.total_pred_areas + self.total_target_areas + self.smooth
        )
        return np.mean(ious), np.mean(dices)

# CORREÇÃO 3: Pesos pré-calculados em amostragem rápida para evitar gargalo de I/O
def get_fast_class_weights(dataset, num_samples=30):
    counts = np.zeros(3, dtype=np.float64)
    step = max(1, len(dataset) // num_samples)

    for idx in range(0, len(dataset), step):
        _, target, _ = dataset[idx]
        target_np = target.numpy()
        for c in range(3):
            counts[c] += np.sum(target_np == c)

    frequencies = counts / counts.sum()
    weights = 1.0 / (frequencies + 1e-6)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)

def train_trilha_a_worst_failures():
    RAW_DIR = Path("data/raw/stage1_train")
    SPLIT_PATH = Path("data/processed/split.json")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    train_ds = DSB2018Dataset(RAW_DIR, splits["train"], transform=get_transforms("train"))
    val_ds   = DSB2018Dataset(RAW_DIR, splits["val"], transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = get_fast_class_weights(train_ds)

    print("Pesos calculados:")
    print(f"Fundo:     {class_weights[0]:.4f}")
    print(f"Interior:  {class_weights[1]:.4f}")
    print(f"Fronteira: {class_weights[2]:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando Trilha A - U-Net (ResNet-18) em: {device}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_iou = 0.0
    best_val_loss = float("inf")
    epochs = 25

    metric_tracker = MetricTracker(num_classes=3)
    start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        # --- TREINO ---
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        metric_tracker.reset()

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                metric_tracker.update(outputs, masks)

        val_loss /= len(val_ds)
        mean_val_iou, mean_val_dice = metric_tracker.compute()

        scheduler.step(mean_val_iou)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(mean_val_iou, 4),
            "val_dice": round(mean_val_dice, 4),
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Época {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {mean_val_iou:.4f} | "
              f"Val Dice: {mean_val_dice:.4f} | Tempo: {time.time() - start_time:.2f}s")

        # --- SALVANDO OS 3 PESOS DE MODELOS ---
        
        # Pesos 1: Melhor IoU
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            path_iou = CHECKPOINT_DIR / "trilha_a_best_iou.pth"
            torch.save(model.state_dict(), path_iou)
            print(f" -> Salvo [Best IoU]: {path_iou.name} ({best_val_iou:.4f})")

        # Pesos 2: Menor Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path_loss = CHECKPOINT_DIR / "trilha_a_best_loss.pth"
            torch.save(model.state_dict(), path_loss)
            print(f" -> Salvo [Best Loss]: {path_loss.name} ({best_val_loss:.4f})")

        # Pesos 3: Última Época (salvo a cada iteração)
        path_last = CHECKPOINT_DIR / "trilha_a_last_epoch.pth"
        torch.save(model.state_dict(), path_last)

    total_train_time = time.time() - start_time

    log_data = {
        "step": "Parte 2 - Trilha A - Treino",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Unet-ResNet18",
        "epochs": epochs,
        "batch_size": 16,
        "total_train_time_seconds": round(total_train_time, 2),
        "best_val_iou": round(best_val_iou, 4),
        "best_val_loss": round(best_val_loss, 4),
        "history": history
    }

    log_json_path = LOG_DIR / "parte2_trilha_a_history_seed2028.json"
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    print(f"\nTreinamento finalizado. Três checkpoints gerados em '{CHECKPOINT_DIR}/'.")

if __name__ == "__main__":
    train_trilha_a_worst_failures()