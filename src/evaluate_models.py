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
RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"

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

def evaluate_specific_model(model, checkpoint_path, output_json_name):
    # Carrega pesos
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    transform = get_val_transform()
    
    with open(SPLIT_PATH, "r") as f:
        val_ids = json.load(f)["val"]

    results = []
    print(f"Avaliando: {checkpoint_path.name}")
    
    with torch.no_grad():
        for sample_id in val_ids:
            sample_folder = RAW_DIR / sample_id
            img_path = list((sample_folder / "images").glob("*.png"))[0]
            orig_img = np.array(Image.open(img_path).convert("RGB"))
            
            pred_masks, _, _, _ = predict_instances(model, orig_img, transform)
            gt_masks = load_ground_truth_instances(sample_folder)
            
            image_map = evaluate_image_ap(pred_masks, gt_masks)
            results.append({"id": sample_id, "gt_count": len(gt_masks), "pred_count": len(pred_masks), "map": float(image_map)})
            
    mean_map = np.mean([r["map"] for r in results])
    
    # Salva o resultado específico deste modelo
    log_data = {"checkpoint": str(checkpoint_path.name), "metrics": {"mAP": round(mean_map, 4)}, "results": results}
    with open(LOG_DIR / output_json_name, "w") as f:
        json.dump(log_data, f, indent=4)
        
    return mean_map, results

def run_all_evaluations():
    models_to_evaluate = [
        # (Arquitetura, Caminho do Peso, Nome do JSON de saída)
        (smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "trilha_a_unet_resnet18_seed2027.pth", "eval_unet_2027.json"),
        (smp.Unet(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "trilha_a_unet_resnet18_seed42.pth", "eval_unet_42.json"),
        (smp.DeepLabV3Plus(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "eixo1_deeplab_resnet18_seed2027.pth", "eval_deeplab_2027.json"),
        (smp.DeepLabV3Plus(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "eixo1_deeplab_resnet18_seed42.pth", "eval_deeplab_42.json"),
        (smp.PSPNet(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "eixo3_pspnet_resnet18_seed2027.pth", "eval_pspnet_2027.json"),
        (smp.PSPNet(encoder_name="resnet18", in_channels=3, classes=3).to(device), 
         ROOT_DIR / "checkpoints" / "eixo3_pspnet_resnet18_seed42.pth", "eval_pspnet_42.json")
    ]

    for model, ckpt, json_out in models_to_evaluate:
        if ckpt.exists():
            evaluate_specific_model(model, ckpt, json_out)
        else:
            print(f"Pulo: {ckpt.name} não encontrado.")

if __name__ == "__main__":
    run_all_evaluations()