import json
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import label
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

LOG_DIR = ROOT_DIR / "results" / "logs"
MOSAIC_DIR = ROOT_DIR / "data" / "processed" / "mosaic"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# --- FUNÇÕES DE MÉTRICA (As mesmas da Parte 2 e 3) ---
def compute_iou_matrix(pred_masks, gt_masks):
    num_preds, num_gts = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)
    for i, p_mask in enumerate(pred_masks):
        for j, g_mask in enumerate(gt_masks):
            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0
    return iou_matrix

def evaluate_image_ap(pred_masks, gt_masks, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    if len(pred_masks) == 0 and len(gt_masks) == 0: return 1.0
    if len(pred_masks) == 0 or len(gt_masks) == 0: return 0.0

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    aps = []
    for t in iou_thresholds:
        matched_gt = set()
        matched_pred = set()
        pairs = [(iou_matrix[i, j], i, j) for i in range(len(pred_masks)) for j in range(len(gt_masks)) if iou_matrix[i, j] >= t]
        pairs.sort(key=lambda x: x[0], reverse=True)

        tp = 0
        for iou, i, j in pairs:
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        denom = tp + fp + fn
        aps.append(float(tp / denom) if denom > 0 else 0.0)
    return np.mean(aps)

# --- INFERÊNCIA DO MODELO ---
def predict_tile(model, tile_img, transform):
    h, w = tile_img.shape[:2]
    augmented = transform(image=tile_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    class_map = np.argmax(probs, axis=0)
    foreground = class_map != 0
    interior = class_map == 1
    
    markers, num_markers = label(interior)
    if num_markers == 0:
        return np.zeros((h, w), dtype=np.int32), 0

    instance_labels = watershed(probs[2], markers=markers, mask=foreground)
    return instance_labels, instance_labels.max()

# --- ALGORITMOS DA PARTE 4 ---
def process_mosaic_naive(model, mosaic_img, transform, tile_size=256):
    """Abordagem ingênua: recorta as imagens, infere e cola de volta (gera quebras)"""
    h, w = mosaic_img.shape[:2]
    naive_mask = np.zeros((h, w), dtype=np.int32)
    
    current_id_offset = 0
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = mosaic_img[y:y+tile_size, x:x+tile_size]
            
            # Pula tiles vazios nas bordas (se houver)
            if tile.shape[0] == 0 or tile.shape[1] == 0: continue
                
            tile_instances, max_id = predict_tile(model, tile, transform)
            
            # Transfere para a máscara global adicionando um offset para evitar IDs iguais
            tile_instances[tile_instances > 0] += current_id_offset
            naive_mask[y:y+tile_size, x:x+tile_size] = tile_instances
            
            current_id_offset += max_id

    return naive_mask

def process_mosaic_fused(model, mosaic_img, transform, tile_size=256, overlap=64):
    """Abordagem de Fusão: Desliza janela com sobreposição e une instâncias que coincidem"""
    h, w = mosaic_img.shape[:2]
    fused_mask = np.zeros((h, w), dtype=np.int32)
    stride = tile_size - overlap
    
    current_id_offset = 0
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = mosaic_img[y:y+tile_size, x:x+tile_size]
            tile_instances, max_id = predict_tile(model, tile, transform)
            
            tile_instances_offset = np.where(tile_instances > 0, tile_instances + current_id_offset, 0)
            
            # Região atual do canvas
            canvas_region = fused_mask[y:y+tile_size, x:x+tile_size]
            
            # Encontra onde o tile atual sobrepõe instâncias já existentes no canvas
            overlap_mask = (canvas_region > 0) & (tile_instances_offset > 0)
            
            if np.any(overlap_mask):
                # Para cada instância do tile atual, ver qual instância do canvas ela mais toca
                unique_new_ids = np.unique(tile_instances_offset[overlap_mask])
                for new_id in unique_new_ids:
                    # Encontra os IDs antigos que essa nova instância está tocando
                    touching_old_ids = canvas_region[(tile_instances_offset == new_id) & overlap_mask]
                    if len(touching_old_ids) > 0:
                        # Pega o ID antigo mais frequente
                        best_old_id = np.bincount(touching_old_ids).argmax()
                        # Atualiza a nova instância para ter o mesmo ID da antiga (FUSÃO)
                        tile_instances_offset[tile_instances_offset == new_id] = best_old_id
            
            # Insere no canvas (dá preferência aos IDs já mesclados ou novos)
            mask_to_update = tile_instances_offset > 0
            fused_mask[y:y+tile_size, x:x+tile_size][mask_to_update] = tile_instances_offset[mask_to_update]
            
            current_id_offset += max_id

    return fused_mask

def main(model_architecture, checkpoint_name):
    # 1. Carrega modelo
    print(f"Carregando modelo {checkpoint_name}...")
    model = model_architecture
    model.load_state_dict(torch.load(ROOT_DIR / "checkpoints" / checkpoint_name, map_location=device))
    model.eval()
    transform = get_val_transform()

    # 2. Carrega o Mosaico
    print("Carregando mosaico e ground truth...")
    mosaic_img = np.array(Image.open(MOSAIC_DIR / "mosaic_image.png").convert("RGB"))
    gt_mask_labels = np.load(MOSAIC_DIR / "mosaic_mask.npy")
    
    # Extrai lista de instâncias do Ground Truth
    gt_masks = [(gt_mask_labels == i) for i in np.unique(gt_mask_labels) if i > 0]

    # 3. Inferência Ingênua
    print("Executando inferência ingênua (sem fusão)...")
    naive_mask_labels = process_mosaic_naive(model, mosaic_img, transform, tile_size=256)
    naive_masks = [(naive_mask_labels == i) for i in np.unique(naive_mask_labels) if i > 0]
    naive_map = evaluate_image_ap(naive_masks, gt_masks)

    # 4. Inferência com Fusão
    print("Executando inferência com fusão (stride + IoU matching)...")
    fused_mask_labels = process_mosaic_fused(model, mosaic_img, transform, tile_size=256, overlap=64)
    fused_masks = [(fused_mask_labels == i) for i in np.unique(fused_mask_labels) if i > 0]
    fused_map = evaluate_image_ap(fused_masks, gt_masks)

    # 5. Resultados
    print("\n=== RESULTADOS DA PARTE 4 (MOSAICO) ===")
    print(f"mAP Ingênuo (Células cortadas): {naive_map:.4f}")
    print(f"mAP com Fusão (Células unidas): {fused_map:.4f}")

    # 6. Salvar visualização
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mosaic_img)
    axes[0].set_title("Mosaico Original")
    axes[0].axis('off')

    # Colorize labels for visualization
    def colorize(mask):
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in np.unique(mask):
            if i > 0:
                colored[mask == i] = np.random.randint(50, 255, size=3)
        return colored

    axes[1].imshow(colorize(naive_mask_labels))
    axes[1].set_title(f"Ingênuo (mAP: {naive_map:.2f})\nNotar cortes secos nas bordas")
    axes[1].axis('off')

    axes[2].imshow(colorize(fused_mask_labels))
    axes[2].set_title(f"Com Fusão (mAP: {fused_map:.2f})\nInstâncias unidas nas bordas")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "mosaic_comparison.png", dpi=300)
    print(f"\nVisualização salva em: {REPORTS_DIR / 'mosaic_comparison.png'}")

if __name__ == "__main__":
    melhor_arquitetura = smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device)
    melhor_peso = "trilha_a_unet_resnet18_seed2028.pth" 
    
    main(melhor_arquitetura, melhor_peso)