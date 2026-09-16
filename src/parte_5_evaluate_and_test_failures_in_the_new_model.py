import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from train_part2_s2 import DSB2018Dataset, get_transforms, set_seed

def compare_worst_failures():
    set_seed(2028)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
    SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"
    
    # Checkpoints
    OLD_CHECKPOINT = ROOT_DIR / "checkpoints" / "trilha_a_unet_resnet18_seed2028.pth"
    NEW_CHECKPOINT = ROOT_DIR / "checkpoints" / "trilha_a_best_iou.pth"
    
    OUTPUT_DIR = ROOT_DIR / "results" / "failure_analysis_comparison"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    val_ds = DSB2018Dataset(RAW_DIR, splits["val"], transform=get_transforms("val"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carrega modelo Antigo
    model_old = smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device)
    model_old.load_state_dict(torch.load(OLD_CHECKPOINT, map_location=device))
    model_old.eval()

    # 2. Carrega modelo Novo (Best IoU)
    model_new = smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device)
    model_new.load_state_dict(torch.load(NEW_CHECKPOINT, map_location=device))
    model_new.eval()

    print("Passo 1: Mapeando os 5 piores casos no modelo antigo...")
    sample_scores = []
    
    with torch.no_grad():
        for images, masks, sample_ids in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model_old(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # IoU médio da amostra
            ious = []
            for c in range(3):
                p_c = (preds == c)
                t_c = (masks == c)
                inter = (p_c & t_c).sum().float()
                union = p_c.sum().float() + t_c.sum().float() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                ious.append(iou.item())

            sample_scores.append({
                "sample_id": sample_ids[0],
                "old_iou": np.mean(ious),
                "image": images[0].cpu().numpy(),
                "gt": masks[0].cpu().numpy(),
                "old_pred": preds[0].cpu().numpy()
            })

    # Seleciona as 5 piores amostras do modelo antigo
    sample_scores.sort(key=lambda x: x["old_iou"])
    worst_5 = sample_scores[:5]

    print("\nPasso 2: Rodando inferência no novo modelo (trilha_a_best_iou.pth)...")
    
    for rank, item in enumerate(worst_5, 1):
        # Prepara a imagem para o modelo novo
        img_tensor = torch.from_numpy(item["image"]).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(item["gt"]).unsqueeze(0).to(device)

        with torch.no_grad():
            output_new = model_new(img_tensor)
            pred_new = torch.argmax(torch.softmax(output_new, dim=1), dim=1)[0]

            # Calcula novo IoU
            ious_new = []
            for c in range(3):
                p_c = (pred_new == c)
                t_c = (mask_tensor[0] == c)
                inter = (p_c & t_c).sum().float()
                union = p_c.sum().float() + t_c.sum().float() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                ious_new.append(iou.item())

            new_iou = np.mean(ious_new)

        # Plot comparativo: Original | Ground Truth | Predição Antiga | Nova Predição
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # Normalização de exibição da imagem RGB
        img_vis = np.transpose(item["image"], (1, 2, 0))
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)

        axes[0].imshow(img_vis)
        axes[0].set_title(f"Imagem Original\nID: {item['sample_id'][:8]}")
        axes[0].axis("off")

        axes[1].imshow(item["gt"], vmin=0, vmax=2, cmap="viridis")
        axes[1].set_title("Ground Truth\n(0:BG, 1:INT, 2:BOUND)")
        axes[1].axis("off")

        axes[2].imshow(item["old_pred"], vmin=0, vmax=2, cmap="viridis")
        axes[2].set_title(f"Modelo Antigo\n(IoU: {item['old_iou']:.4f})")
        axes[2].axis("off")

        axes[3].imshow(pred_new.cpu().numpy(), vmin=0, vmax=2, cmap="viridis")
        axes[3].set_title(f"Novo Modelo (Best IoU)\n(IoU: {new_iou:.4f})")
        axes[3].axis("off")

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"comparativo_rank_{rank}_{item['sample_id'][:8]}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        delta_iou = new_iou - item['old_iou']
        sinal = "+" if delta_iou >= 0 else ""
        print(f"Rank {rank} [ID {item['sample_id'][:8]}]: IoU Antigo = {item['old_iou']:.4f} -> Novo IoU = {new_iou:.4f} ({sinal}{delta_iou:.4f})")
        print(f" -> Imagem salva em: {save_path}")

if __name__ == "__main__":
    compare_worst_failures()