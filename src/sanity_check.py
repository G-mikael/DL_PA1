import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset

import time
import numpy as np
import cv2
import random
from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import login


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

dotenv_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Token do Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

class SyntheticDataset(Dataset):
    def __init__(self, data_dir="data/synthetic"):
        self.img_paths = sorted(list(Path(data_dir).glob("images/*.png")))
        self.mask_paths = sorted(list(Path(data_dir).glob("masks/*.npy")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Lê a imagem RGB e converte para tensor
        img = cv2.cvtColor(cv2.imread(str(self.img_paths[idx])), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Lê a máscara e converte para tensor binário
        mask = np.load(str(self.mask_paths[idx]))
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor

def run_sanity_check():
    # Carrega 2 amostras do conjunto gerado
    full_dataset = SyntheticDataset(data_dir="data/synthetic")
    sanity_dataset = Subset(full_dataset, indices=[0, 1])
    sanity_loader = DataLoader(sanity_dataset, batch_size=2, shuffle=False)

    # Inicializa Modelo Pré-treinado (ResNet-18)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Executando no dispositivo: {device}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # Otimizador (adam) e Loss (BCE com Logits)
    # Adam escolhido pela velocidade de convergência e tendencia a não ficar preso em mínimos locais
    # BCEWithLogitsLoss combina sigmoid + BCE, evitando logs próximos de zero, e útil para problemas de segmentação binária
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Loop de Overfitting (150 épocas)
    model.train()
    print("Iniciando Overfitting Sanity Check...")
    start_time = time.time()
    
    for epoch in range(1, 151):
        for images, masks in sanity_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        if epoch % 30 == 0 or epoch == 1:
            print(f"Época {epoch:03d}/150 | Loss: {loss.item():.6f} | Tempo decorrido: {time.time() - start_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\nTempo total de treino: {total_time:.2f}s")    
    
    # Validação do Critério de Parada
    if loss.item() < 0.05:
        print(f"\n SANITY CHECK APROVADO: A loss convergiu para {loss.item():.6f}!")
        print("O pipeline de tensores, modelo e retropropagação está funcional.")
    else:
        print(f"\n SANITY CHECK FALHOU: A loss não caiu suficientemente. Valor final: {loss.item():.6f}")
        print("Verifique learning rate, arquitetura do modelo ou dimensões dos tensores ou cálculo da loss.")

if __name__ == "__main__":
    run_sanity_check()