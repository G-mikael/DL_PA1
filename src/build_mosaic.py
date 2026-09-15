import json
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "stage1_train"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split.json"

def load_and_resize_sample(sample_id, target_size=(256, 256)):
    sample_folder = RAW_DIR / sample_id
    
    # Carrega e redimensiona a imagem
    img_path = list((sample_folder / "images").glob("*.png"))[0]
    img = np.array(Image.open(img_path).convert("RGB"))
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Carrega, une e redimensiona as máscaras (Ground Truth)
    mask_paths = list((sample_folder / "masks").glob("*.png"))
    merged_mask = np.zeros(img.shape[:2], dtype=np.int32)
    
    for i, m_path in enumerate(mask_paths, start=1):
        m = np.array(Image.open(m_path)) > 0
        merged_mask[m] = i
        
    mask_resized = cv2.resize(merged_mask, target_size, interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized

def build_mosaic():
    with open(SPLIT_PATH, "r") as f:
        val_ids = json.load(f)["val"]
        
    # Escolhe 4 imagens aleatórias (ou fixas) da validação
    selected_ids = val_ids[:4] 
    
    # Carrega os 4 quadrantes
    img1, mask1 = load_and_resize_sample(selected_ids[0])
    img2, mask2 = load_and_resize_sample(selected_ids[1])
    img3, mask3 = load_and_resize_sample(selected_ids[2])
    img4, mask4 = load_and_resize_sample(selected_ids[3])
    
    # Para as máscaras não terem IDs repetidos de instâncias entre as imagens:
    mask2[mask2 > 0] += mask1.max()
    mask3[mask3 > 0] += mask2.max()
    mask4[mask4 > 0] += mask3.max()

    # Concatenação (Grid 2x2)
    top_row_img = np.hstack((img1, img2))
    bottom_row_img = np.hstack((img3, img4))
    mosaic_img = np.vstack((top_row_img, bottom_row_img))
    
    top_row_mask = np.hstack((mask1, mask2))
    bottom_row_mask = np.hstack((mask3, mask4))
    mosaic_mask = np.vstack((top_row_mask, bottom_row_mask))
    
    # Salva os resultados
    output_dir = ROOT_DIR / "data" / "processed" / "mosaic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(mosaic_img).save(output_dir / "mosaic_image.png")
    # Salva a máscara como npy pois contém IDs de instâncias (>255)
    np.save(output_dir / "mosaic_mask.npy", mosaic_mask)
    
    print(f"Mosaico gerado com sucesso em: {output_dir}")
    print(f"Tamanho do mosaico: {mosaic_img.shape}")

if __name__ == "__main__":
    build_mosaic()