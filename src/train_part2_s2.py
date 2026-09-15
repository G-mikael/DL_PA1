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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2028)

# Augmentations via Albumentations
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

def build_three_class_target(mask_paths, h, w, erosion_size=2):
    target = np.zeros((h, w), dtype=np.uint8)

    all_interiors = np.zeros((h, w), dtype=bool)
    all_boundaries = np.zeros((h, w), dtype=bool)

    for m_path in mask_paths:
        mask = np.array(Image.open(m_path)) > 0

        interior = binary_erosion(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=erosion_size
        )

        boundary = mask & ~interior

        all_interiors |= interior
        all_boundaries |= boundary

    # 1 = interior
    target[all_interiors] = 1

    # 2 = boundary (tem prioridade)
    target[all_boundaries] = 2

    return target

# PyTorch Dataset para o DSB2018
class DSB2018Dataset(Dataset):
    def __init__(self, raw_dir, sample_ids, transform=None):
        self.raw_dir = Path(raw_dir)
        self.sample_ids = sample_ids
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_folder = self.raw_dir / self.sample_ids[idx]

        # Imagem
        img_path = list((sample_folder / "images").glob("*.png"))[0]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Máscaras das instâncias
        mask_paths = list((sample_folder / "masks").glob("*.png"))
        h, w = image.shape[:2]

        # Target da Trilha A:
        # 0 = fundo
        # 1 = interior
        # 2 = fronteira
        target = build_three_class_target(
            mask_paths,
            h,
            w,
            erosion_size=2
        )

        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"].long()

        return image, target, self.sample_ids[idx]

# Métricas de Avaliação Semântica (IoU e Dice)
def calculate_semantic_metrics(logits, targets, smooth=1e-6):
    preds = logits.argmax(dim=1)

    ious = []
    dices = []

    for c in range(3):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum(dim=(1, 2)).float()
        pred_area = pred_c.sum(dim=(1, 2)).float()
        target_area = target_c.sum(dim=(1, 2)).float()

        union = pred_area + target_area - intersection

        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (
            pred_area + target_area + smooth
        )

        ious.append(iou.mean().item())
        dices.append(dice.mean().item())

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    return mean_iou, mean_dice

def calculate_class_weights(dataset):
    counts = np.zeros(3, dtype=np.float64)

    for idx in range(len(dataset)):
        _, target, _ = dataset[idx]

        target = target.numpy()

        for c in range(3):
            counts[c] += np.sum(target == c)

    frequencies = counts / counts.sum()

    weights = 1.0 / frequencies
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)

# Pipeline de Treinamento e Validação
def train_trilha_a():
    RAW_DIR = Path("data/raw/stage1_train")
    SPLIT_PATH = Path("data/processed/split.json")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    # Carregamento dos DataLoaders
    train_ds = DSB2018Dataset(RAW_DIR, splits["train"], transform=get_transforms("train"))
    val_ds   = DSB2018Dataset(RAW_DIR, splits["val"], transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = calculate_class_weights(train_ds)

    print("Pesos das classes:")
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
    epochs = 25

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
        val_ious, val_dices = [], []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
                iou, dice = calculate_semantic_metrics(outputs, masks)
                val_ious.append(iou)
                val_dices.append(dice)

        val_loss /= len(val_ds)
        mean_val_iou = np.mean(val_ious)
        mean_val_dice = np.mean(val_dices)

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
              f"Val Dice: {mean_val_dice:.4f} | Tempo decorrido: {time.time() - start_time:.2f}s")

        # Salva o melhor checkpoint baseado no IoU semântico
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            checkpoint_path = CHECKPOINT_DIR / "trilha_a_unet_resnet18_seed2028.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Novo melhor modelo salvo em: {checkpoint_path} (IoU: {best_val_iou:.4f})")
    
    total_train_time = time.time() - start_time

    #Logging do histórico de treino e métricas
    log_data = {
        "step": "Parte 2 - Trilha A - Treino",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Unet-ResNet18",
        "epochs": epochs,
        "batch_size": 16,
        "lr_initial": 1e-3,
        "total_train_time_seconds": round(total_train_time, 2),
        "best_val_iou": round(best_val_iou, 4),
        "history": history
    }

    log_json_path = LOG_DIR / "parte2_trilha_a_history_seed2028.json"
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    log_txt_path = LOG_DIR / "parte2_trilha_a_history_seed2028.txt"
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write("=== LOG PARTE 2: TREINO TRILHA A ===\n")
        f.write(f"Data: {log_data['timestamp']}\n")
        f.write(f"Tempo Total de Treino: {log_data['total_train_time_seconds']}s\n")
        f.write(f"Melhor Val IoU: {log_data['best_val_iou']}\n")

    print(f"Histórico e logs salvos em: '{log_json_path}' e '{log_txt_path}'")

if __name__ == "__main__":
    train_trilha_a()