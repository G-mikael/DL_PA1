import cv2
import numpy as np
import os
from pathlib import Path
import random

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)


def generate_ellipse_dataset(num_samples=100, output_dir="data/synthetic", img_size=(128, 128), seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    for idx in range(num_samples):
        # Fundo base com variação aleatória
        bg_color = np.random.randint(20, 60)
        image = np.full((*img_size, 3), bg_color, dtype=np.uint8)
        
        # Array com máscara de instâncias com id único para cada elipse
        instance_mask = np.zeros(img_size, dtype=np.int32)
        num_ellipses = np.random.randint(5, 21)
        
        for inst_id in range(1, num_ellipses + 1):
            # Coordenadas do centro, eixos e ângulos aleatórios
            center = (np.random.randint(15, img_size[1] - 15), 
                      np.random.randint(15, img_size[0] - 15))
            axes = (np.random.randint(8, 22), np.random.randint(5, 15))
            angle = np.random.randint(0, 180)
            
            # Tom de cinza para a elipse
            obj_color = np.random.randint(120, 240)

            # Desenha a elipse na imagem RGB
            cv2.ellipse(image, center, axes, angle, 0, 360, (obj_color, obj_color, obj_color), -1) #-1 remove a borda
            
            # Desenha o ID de instância na máscara, sobreposição implica substituição do ID anterior
            temp_mask = np.zeros(img_size, dtype=np.uint8)
            cv2.ellipse(temp_mask, center, axes, angle, 0, 360, 255, -1)
            instance_mask[temp_mask == 255] = inst_id

        # Ruído Gaussiano
        noise = np.random.normal(0, np.random.uniform(5, 15), image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Salvar
        cv2.imwrite(f"{output_dir}/images/sample_{idx:04d}.png", noisy_image)
        np.save(f"{output_dir}/masks/sample_{idx:04d}.npy", instance_mask)

    print(f"{num_samples} amostras sintéticas geradas com sucesso em '{output_dir}'.")

if __name__ == "__main__":
    generate_ellipse_dataset(num_samples=100)