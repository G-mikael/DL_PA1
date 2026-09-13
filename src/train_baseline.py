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
from pathlib import Path
import os

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

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
        
        # Leitura da imagem
        img_path = list((sample_folder / "images").glob("*.png"))[0]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Consolidação de múltiplas máscaras de instância em uma única máscara binária semântica
        mask_paths = list((sample_folder / "masks").glob("*.png"))
        h, w = image.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.float32)

        for m_path in mask_paths:
            m = np.array(Image.open(m_path))
            semantic_mask = np.maximum(semantic_mask, (m > 0).astype(np.float32))

        if self.transform:
            augmented = self.transform(image=image, mask=semantic_mask)
            image = augmented["image"]
            semantic_mask = augmented["mask"].unsqueeze(0)  # Shape: (1, H, W)

        return image, semantic_mask, self.sample_ids[idx]

# Métricas de Avaliação Semântica (IoU e Dice)
def calculate_semantic_metrics(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    total_pixels = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    union = total_pixels - intersection

    iou = ((intersection + smooth) / (union + smooth)).mean().item()
    dice = ((2.0 * intersection + smooth) / (total_pixels + smooth)).mean().item()
    return iou, dice

# Pipeline de Treinamento e Validação
def train_baseline():
    RAW_DIR = Path("data/raw/stage1_train")
    SPLIT_PATH = Path("data/processed/split.json")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    # Carregamento dos DataLoaders
    train_ds = DSB2018Dataset(RAW_DIR, splits["train"], transform=get_transforms("train"))
    val_ds   = DSB2018Dataset(RAW_DIR, splits["val"], transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando Baseline U-Net (ResNet-18) em: {device}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_iou = 0.0
    epochs = 40

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
                probs = torch.sigmoid(outputs)
                
                iou, dice = calculate_semantic_metrics(probs, masks)
                val_ious.append(iou)
                val_dices.append(dice)

        val_loss /= len(val_ds)
        mean_val_iou = np.mean(val_ious)
        mean_val_dice = np.mean(val_dices)

        scheduler.step(mean_val_iou)

        print(f"Época {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {mean_val_iou:.4f} | "
              f"Val Dice: {mean_val_dice:.4f}")

        # Salva o melhor checkpoint baseado no IoU semântico
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            checkpoint_path = CHECKPOINT_DIR / "baseline_unet_resnet18.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Novo melhor modelo salvo em: {checkpoint_path} (IoU: {best_val_iou:.4f})")

if __name__ == "__main__":
    train_baseline()