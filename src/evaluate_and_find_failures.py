import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from train_part2_s2 import DSB2018Dataset, get_transforms, set_seed

def find_worst_failures():
    set_seed(2028)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
    SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"
    CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "trilha_a_unet_resnet18_seed2028.pth"
    OUTPUT_DIR = ROOT_DIR / "results" / "failure_analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    val_ds = DSB2018Dataset(RAW_DIR, splits["val"], transform=get_transforms("val"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    sample_scores = []

    print("Avaliando amostras individuais para encontrar as 5 piores predições...")
    with torch.no_grad():
        for images, masks, sample_ids in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Cálculo de IoU mean na amostra
            ious = []
            for c in range(3):
                p_c = (preds == c)
                t_c = (masks == c)
                inter = (p_c & t_c).sum().float()
                union = p_c.sum().float() + t_c.sum().float() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                ious.append(iou.item())

            sample_iou = np.mean(ious)
            sample_scores.append({
                "sample_id": sample_ids[0],
                "iou": sample_iou,
                "image": images[0].cpu().numpy(),
                "gt": masks[0].cpu().numpy(),
                "pred": preds[0].cpu().numpy(),
                "boundary_map": probs[0, 2].cpu().numpy() # Mapa intermediário de probabilidade da Fronteira
            })

    # Ordena amostras pelo menor IoU
    sample_scores.sort(key=lambda x: x["iou"])
    worst_5 = sample_scores[:5]

    # Gera visualização das 5 piores falhas
    for rank, item in enumerate(worst_5, 1):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Desnormalização simples para exibição da imagem
        img_vis = np.transpose(item["image"], (1, 2, 0))
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)

        axes[0].imshow(img_vis)
        axes[0].set_title(f"Imagem Original\nID: {item['sample_id'][:8]}")
        axes[0].axis("off")

        axes[1].imshow(item["gt"], vmin=0, vmax=2, cmap="viridis")
        axes[1].set_title("Ground Truth\n(0:BG, 1:INT, 2:BOUND)")
        axes[1].axis("off")

        axes[2].imshow(item["pred"], vmin=0, vmax=2, cmap="viridis")
        axes[2].set_title(f"Predição Final\n(IoU Amostra: {item['iou']:.4f})")
        axes[2].axis("off")

        im = axes[3].imshow(item["boundary_map"], cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Mapa Intermediário\nProbabilidade de Fronteira")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"falha_rank_{rank}_{item['sample_id'][:8]}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Figura de falha saved em: {save_path}")

if __name__ == "__main__":
    find_worst_failures()