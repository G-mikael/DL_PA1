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
import time

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

#CONFIGURAÇÕES DE DADOS E DISPOSITIVO
LOG_DIR = ROOT_DIR / "results" / "logs"
LOG_DIR.mkdir(exist_ok=True)
RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "trilha_a_unet_resnet18.pth"

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
        
        denom = tp + fp + fn
        ap = float(tp / denom) if denom > 0 else 0.0
        aps.append(ap)

    return np.mean(aps) # mAP da imagem sobre os 10 limiares

def predict_instances(model, orig_img, transform):
    h, w = orig_img.shape[:2]

    # Pré-processamento
    augmented = transform(image=orig_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Inferência
    output = model(input_tensor)

    # Probabilidades das 3 classes
    probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    # Volta para o tamanho original
    background_prob = cv2.resize(
        probs[0], (w, h), interpolation=cv2.INTER_LINEAR
    )
    interior_prob = cv2.resize(
        probs[1], (w, h), interpolation=cv2.INTER_LINEAR
    )
    boundary_prob = cv2.resize(
        probs[2], (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Classe mais provável
    class_map = np.argmax(
        np.stack([
            background_prob,
            interior_prob,
            boundary_prob
        ], axis=0),
        axis=0
    )

    # Foreground e interiores
    foreground = class_map != 0
    interior = class_map == 1

    # Marcadores
    markers, num_markers = label(interior)

    # Caso não exista nenhum marker
    if num_markers == 0:
        instance_labels = np.zeros(
            (h, w),
            dtype=np.int32
        )

        return [], class_map, instance_labels, num_markers

    # Watershed
    instance_labels = watershed(
        boundary_prob,
        markers=markers,
        mask=foreground
    )

    # Instâncias finais
    num_instances = instance_labels.max()

    pred_masks = [
        (instance_labels == i)
        for i in range(1, num_instances + 1)
    ]

    return pred_masks, class_map, instance_labels, num_markers

# Pipeline de avaliação
def run_evaluation():
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)
    val_ids = splits["val"]

    # Carrega modelo
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    transform = get_val_transform()
    
    results = [] # Armazena: (gt_count, map_score, count_error)

    print("Avaliando instâncias e executando Matching Guloso...")
    with torch.no_grad():
        for sample_id in val_ids:
            sample_folder = RAW_DIR / sample_id
            image_folder = sample_folder / "images"

            img_files = list(image_folder.glob("*.png")) + \
                        list(image_folder.glob("*.PNG"))

            if not img_files:
                raise FileNotFoundError(
                    f"\n[ERRO DE CAMINHO] Nenhuma imagem encontrada!\n"
                    f"Caminho testado: {image_folder.resolve()}\n"
                    f"Existe a pasta da amostra? {sample_folder.exists()}\n"
                    f"Existe a pasta 'images'? {image_folder.exists()}"
                )

            img_path = img_files[0]

            orig_img = np.array(
                Image.open(img_path).convert("RGB")
            )

            # Trilha A: 3 classes → markers → watershed
            pred_masks, class_map, instance_labels, num_markers = predict_instances(
                model,
                orig_img,
                transform
            )

            gt_masks = load_ground_truth_instances(sample_folder)

            # Métricas
            image_map = evaluate_image_ap(
                pred_masks,
                gt_masks
            )

            raw_count_error = len(pred_masks) - len(gt_masks)

            results.append({
                "id": sample_id,
                "gt_count": len(gt_masks),
                "pred_count": len(pred_masks),
                "num_markers": int(num_markers),
                "map": float(image_map),
                "count_error": abs(raw_count_error),
                "raw_count_error": raw_count_error
            })

    # --- RELATÓRIO DE MÉTRICAS ---
    mean_map = np.mean([r["map"] for r in results])
    errors = [r["pred_count"] - r["gt_count"] for r in results]
    mean_mae = float(np.mean([abs(e) for e in errors]))
    mean_rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
    
    print("\n=== RESULTADOS DA PARTE 2 - TRILHA A ===")
    print(f"mAP de Instância (IoU 0.50:0.95): {mean_map:.4f}")
    print(f"Erro Médio Absoluto de Contagem (MAE): {mean_mae:.2f} núcleos")
    print(f"Erro Quadrático Médio de Contagem (RMSE): {mean_rmse:.2f} núcleos")

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
    output_graph = (ROOT_DIR / "reports" / "figures" / "trilha_a_evaluation.png")
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de quantificação do fracasso salvo em: {output_graph}")
    
    # Salvamento de Logs Finais
    log_data = {
        "step": "Parte 2 - Avaliação Trilha A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_used": str(CHECKPOINT_PATH),
        "num_val_samples": len(val_ids),
        "metrics": {
            "mAP_instance_050_095": round(mean_map, 4),
            "MAE_count": round(mean_mae, 2),
            "RMSE_count": round(mean_rmse, 2)
        },
        "per_image_results": results
    }

    log_json_path = LOG_DIR / "trilha_a_evaluation_metrics.json"
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    log_txt_path = LOG_DIR / "trilha_a_evaluation_metrics.txt"
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write("=== LOG PARTE 2: AVALIAÇÃO TRILHA A ===\n")
        f.write(f"Data: {log_data['timestamp']}\n")
        f.write(f"Amostras Avaliadas: {log_data['num_val_samples']}\n")
        f.write(f"mAP de Instância (IoU 0.50:0.95): {log_data['metrics']['mAP_instance_050_095']}\n")
        f.write(f"Erro Médio Absoluto (MAE): {log_data['metrics']['MAE_count']}\n")
        f.write(f"RMSE de Contagem: {log_data['metrics']['RMSE_count']}\n")

    print(f"Logs salvos com sucesso em: '{log_json_path}' e '{log_txt_path}'")

if __name__ == "__main__":
    run_evaluation()