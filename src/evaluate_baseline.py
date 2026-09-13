import json
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

#CONFIGURAÇÕES DE DADOS E DISPOSITIVO
RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "baseline_unet_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Carregamento das instâncias de máscara de verdade do solo (ground truth) para uma amostra
def load_ground_truth_instances(sample_folder):
    mask_paths = list((sample_folder / "masks").glob("*.png"))
    gt_masks = []
    for m_path in mask_paths:
        m = np.array(Image.open(m_path)) > 0
        gt_masks.append(m)
    return gt_masks # Lista de máscaras binárias

# Matching guloso
def compute_iou_matrix(pred_masks, gt_masks):
    num_preds = len(pred_masks)
    num_gts = len(gt_masks)
    iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)
    
    for i, p_mask in enumerate(pred_masks):
        for j, g_mask in enumerate(gt_masks):
            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0
            
    return iou_matrix

def evaluate_image_ap(pred_masks, gt_masks, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 1.0
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    aps = []

    for t in iou_thresholds:
        matched_gt = set()
        matched_pred = set()
        
        # Ordena correspondências por IoU decrescente
        pairs = []
        for i in range(len(pred_masks)):
            for j in range(len(gt_masks)):
                if iou_matrix[i, j] >= t:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        tp = 0
        for iou, i, j in pairs:
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1

        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        
        ap = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        aps.append(ap)

    return np.mean(aps) # mAP da imagem sobre os 10 limiares

# Pipeline de avaliação
def run_evaluation():
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)
    val_ids = splits["val"]

    # Carrega modelo
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    transform = get_val_transform()
    
    results = [] # Armazena: (gt_count, map_score, count_error)

    print("Avaliando instâncias e executando Matching Guloso...")
    with torch.no_grad():
        for sample_id in val_ids:
            sample_folder = RAW_DIR / sample_id
            image_folder = sample_folder / "images"
            
            print(f"[DEBUG] Buscando imagens em: {image_folder.resolve()}")
            
            img_files = list(image_folder.glob("*.png")) + list(image_folder.glob("*.PNG"))
            
            if not img_files:
                raise FileNotFoundError(
                    f"\n[ERRO DE CAMINHO] Nenhuma imagem encontrada!\n"
                    f"Caminho testado: {image_folder.resolve()}\n"
                    f"Existe a pasta da amostra? {sample_folder.exists()}\n"
                    f"Existe a pasta 'images'? {image_folder.exists()}"
                )
            
            img_path = img_files[0]
            
            orig_img = np.array(Image.open(img_path).convert("RGB"))
            h, w = orig_img.shape[:2]

            # Inferência
            augmented = transform(image=orig_img)
            input_tensor = augmented["image"].unsqueeze(0).to(device)
            output = model(input_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

            # Redimensiona probabilidade para o tamanho original da imagem
            prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Pós processamento ingênuo
            binary_mask = prob_full > 0.5
            labeled_mask, num_features = label(binary_mask)
            
            pred_masks = [(labeled_mask == i) for i in range(1, num_features + 1)]
            gt_masks = load_ground_truth_instances(sample_folder)

            # Métricas por imagem
            image_map = evaluate_image_ap(pred_masks, gt_masks)
            count_error = abs(len(pred_masks) - len(gt_masks))
            
            results.append({
                "id": sample_id,
                "gt_count": len(gt_masks),
                "pred_count": len(pred_masks),
                "map": image_map,
                "count_error": count_error
            })

    # --- RELATÓRIO DE MÉTRICAS ---
    mean_map = np.mean([r["map"] for r in results])
    mean_mae = np.mean([r["count_error"] for r in results])
    
    print("\n=== RESULTADOS DA PARTE 1 (BASELINE INGÊNUA) ===")
    print(f"mAP de Instância (IoU 0.50:0.95): {mean_map:.4f}")
    print(f"Erro Médio Absoluto de Contagem (MAE): {mean_mae:.2f} núcleos")

    # --- GERAÇÃO DO GRÁFICO DE FRACASSO ---
    gt_counts = [r["gt_count"] for r in results]
    maps = [r["map"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.scatter(gt_counts, maps, alpha=0.7, color='crimson')
    plt.title("Quantificação do Fracasso: mAP vs Densidade de Objetos (Baseline)")
    plt.xlabel("Número de Instâncias Reais (Densidade de Objetos)")
    plt.ylabel("mAP de Instância (IoU 0.50:0.95)")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Salva o gráfico
    output_graph = ROOT_DIR / "reports" / "figures" / "parte1_failure_quantification.png"
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de quantificação do fracasso salvo em: {output_graph}")

if __name__ == "__main__":
    run_evaluation()