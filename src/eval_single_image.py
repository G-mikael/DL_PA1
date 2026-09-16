import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import label
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Resolve caminhos do projeto
CURRENT_FILE = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
SRC_DIR = CURRENT_FILE.parent if CURRENT_FILE.name != "src" else CURRENT_FILE
ROOT_DIR = SRC_DIR.parent if (SRC_DIR.parent / "data").exists() else SRC_DIR
os.chdir(ROOT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_model(checkpoint_path, device_to_use=None):
    if device_to_use is None:
        device_to_use = device

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device_to_use)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=3
    ).to(device_to_use)

    model.load_state_dict(state_dict)
    model.eval()
    return model

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

        pairs = []
        for i in range(len(pred_masks)):
            for j in range(len(gt_masks)):
                if iou_matrix[i, j] >= t:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        tp = 0
        for iou_val, i, j in pairs:
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1

        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp

        denom = tp + fp + fn
        ap = float(tp / denom) if denom > 0 else 0.0
        aps.append(ap)

    return float(np.mean(aps))

def predict_instances(model, orig_img, transform):
    h, w = orig_img.shape[:2]

    augmented = transform(image=orig_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    background_prob = cv2.resize(probs[0], (w, h), interpolation=cv2.INTER_LINEAR)
    interior_prob = cv2.resize(probs[1], (w, h), interpolation=cv2.INTER_LINEAR)
    boundary_prob = cv2.resize(probs[2], (w, h), interpolation=cv2.INTER_LINEAR)

    class_map = np.argmax(
        np.stack([background_prob, interior_prob, boundary_prob], axis=0),
        axis=0
    )

    foreground = class_map != 0
    interior = class_map == 1

    markers, num_markers = label(interior)

    if num_markers == 0:
        instance_labels = np.zeros((h, w), dtype=np.int32)
        return [], class_map, instance_labels, num_markers

    instance_labels = watershed(
        boundary_prob,
        markers=markers,
        mask=foreground
    )

    num_instances = instance_labels.max()
    pred_masks = [(instance_labels == i) for i in range(1, num_instances + 1)]

    return pred_masks, class_map, instance_labels, num_markers

def load_ground_truth_instances(sample_folder):
    sample_folder = Path(sample_folder)
    
    # Tratamento de resguardo caso venha terminado em 'images' ou 'masks'
    if sample_folder.name in ["images", "masks"]:
        sample_folder = sample_folder.parent

    mask_paths = list((sample_folder / "masks").glob("*.png"))
    gt_masks = []
    
    for m_path in mask_paths:
        m = np.array(Image.open(m_path)) > 0
        gt_masks.append(m)
        
    return gt_masks


def process_single_image(sample_code, model, raw_dir=None, device_to_use=None):
    if device_to_use is None:
        device_to_use = device

    # 1. Normalização do caminho base da amostra
    sample_path = Path(sample_code).resolve()
    
    # Se o usuário passou um caminho que aponta para 'images' ou 'masks' (ou dentro delas)
    if "images" in sample_path.parts or "masks" in sample_path.parts:
        while sample_path.name in ["images", "masks"] or sample_path.parent.name in ["images", "masks"]:
            sample_path = sample_path.parent
        sample_folder = sample_path
    elif sample_path.exists():
        sample_folder = sample_path
    else:
        if raw_dir is None:
            raw_dir = ROOT_DIR / "data" / "raw" / "stage1_train"
        # Limpa eventuais sufixos de texto caso tenha sido passada uma string
        clean_code = str(sample_code).split("/images")[0].split("\\images")[0]
        sample_folder = Path(raw_dir) / clean_code

    sample_id_str = sample_folder.name

    # 2. Busca da imagem original (pasta: sample_folder / images)
    img_dir = sample_folder / "images"
    img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")) if img_dir.exists() else []

    if not img_files:
        raise FileNotFoundError(
            f"\n[ERRO DE CAMINHO] Nenhuma imagem encontrada!\n"
            f"Caminho buscado: {img_dir.resolve()}\n"
            f"A pasta da amostra existe? {sample_folder.exists()}\n"
            f"A subpasta 'images' existe? {img_dir.exists()}"
        )

    orig_img = np.array(Image.open(img_files[0]).convert("RGB"))
    h, w = orig_img.shape[:2]

    transform = get_val_transform()

    # 3. Predição e Pós-processamento
    pred_masks, class_map, instance_labels, num_markers = predict_instances(model, orig_img, transform)

    # 4. Busca das máscaras de Ground Truth (pasta: sample_folder / masks)
    gt_masks = load_ground_truth_instances(sample_folder)
    num_gt = len(gt_masks)
    num_preds = len(pred_masks)

    image_map = evaluate_image_ap(pred_masks, gt_masks)

    # 5. Construção dos mapas de exibição
    gt_instances_map = np.zeros((h, w), dtype=np.int32)
    for idx, g_mask in enumerate(gt_masks, start=1):
        gt_instances_map[g_mask] = idx

    # 6. Renderização gráfica (3 painéis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

    # Original
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Imagem Original\nID: {sample_id_str[:12]}...", fontsize=11)
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(np.zeros((h, w, 3), dtype=np.uint8))
    if num_gt > 0:
        gt_masked = np.ma.masked_where(gt_instances_map == 0, gt_instances_map)
        axes[1].imshow(gt_masked, cmap='tab20', interpolation='none')
    axes[1].set_title(f"Ground Truth ({num_gt} instâncias)", fontsize=11)
    axes[1].axis("off")

    # Predição
    axes[2].imshow(np.zeros((h, w, 3), dtype=np.uint8))
    if num_preds > 0:
        pred_masked = np.ma.masked_where(instance_labels == 0, instance_labels)
        axes[2].imshow(pred_masked, cmap='tab20', interpolation='none')
    axes[2].set_title(f"Predição ({num_preds} instâncias)\nmAP (0.50:0.95): {image_map:.4f}", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "sample_id": sample_id_str,
        "num_gt": num_gt,
        "num_preds": num_preds,
        "map_score": round(image_map, 4)
    }

def predict_instance_mask(img_path, model):
    img_path = Path(img_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Erro: Imagem não encontrada no caminho: {img_path}")

    # 1. Carrega a imagem
    orig_img = np.array(Image.open(img_path).convert("RGB"))
    
    # 2. Prepara a transformação definida no arquivo
    transform = get_val_transform()
    
    # 3. Executa o forward pass e o pós-processamento (Watershed) reutilizando sua função interna
    pred_masks, class_map, instance_labels, num_markers = predict_instances(model, orig_img, transform)
    
    # Retorna apenas a matriz 2D final com os IDs das instâncias
    return instance_labels