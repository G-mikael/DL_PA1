import os
import glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import json
import random

import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count()) #só pra remover warning de "loky" ao usar joblib com sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)
DATA_DIR = Path("data/raw/stage1_train")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pastas de amostras
sample_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]

random.seed(2026)
np.random.seed(2026)

def extrair_features(sample_dir):
    """Extrai média, desvio padrão e histograma de intensidade da imagem."""
    img_path = list((sample_dir / "images").glob("*.png"))[0]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Média e desvio padrão por canal de cor (R, G, B)
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))
    
    # Histograma em escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
    
    return np.hstack([mean_rgb, std_rgb, hist]), img_path

print("Extraindo características das imagens...")

features_list = []
img_paths_list = []
print(f"Total de amostras encontradas: {len(sample_dirs)}")

for sample in sample_dirs:
    feat, img_path = extrair_features(sample)
    features_list.append(feat)
    img_paths_list.append(img_path)

print(features_list)
print(img_paths_list)

X = np.array(features_list)

# Buscando separar as imagens em 5 diferentes modalidades usando K-Means
# Modalidade 0 — Fluorescência de Baixa Densidade / Baixo Contraste
# Características: Fundo completamente escuro/preto com núcleos claros e esparsos. É o grupo majoritário do dataset.
# Modalidade 1 — Histologia de Tecido (H&E / Coloração Roxa)
# Características: Lâminas histológicas com coloração roxa/azulada contínua e fundo roxo-claro/cinza.
# Modalidade 2 — Campo Claro / Citologia (Células Isoladas)
# Características: Fundo muito claro/branco com núcleos escuros bem delimitados dentro do citoplasma celular (aparência de citologia em meio líquido).
# Modalidade 3 — Fluorescência de Alta Densidade / Alto Brilho
# Características: Fundo escuro, mas com altíssima densidade de núcleos pequenos e muito brilhantes (alto contraste/saturação).
# Modalidade 4 — Histologia de Núcleos Agrupados (H&E / Fundo Claro)
# Características: Núcleos agrupados em áreas de fundo claro, com coloração característica da coloração H&E.

N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=2026, n_init=10)
cluster_labels = kmeans.fit_predict(X)

print(f"Clustering concluído em {N_CLUSTERS} modalidades.")

# Gerar e Salvar a Imagem Ilustrativa dos Clusters
SAMPLES_PER_CLUSTER = 4
fig, axes = plt.subplots(N_CLUSTERS, SAMPLES_PER_CLUSTER, figsize=(12, 3 * N_CLUSTERS))

if N_CLUSTERS == 1:
    axes = np.expand_dims(axes, axis=0)

for c in range(N_CLUSTERS):
    # Pega os caminhos das imagens pertencentes ao cluster c
    idxs_cluster = np.where(cluster_labels == c)[0]
    # Seleciona até SAMPLES_PER_CLUSTER amostras aleatórias do cluster
    chosen_idxs = np.random.choice(idxs_cluster, size=min(SAMPLES_PER_CLUSTER, len(idxs_cluster)), replace=False)
    
    for idx_col, img_idx in enumerate(chosen_idxs):
        ax = axes[c, idx_col]
        img = Image.open(img_paths_list[img_idx])
        ax.imshow(img)
        ax.axis("off")
        if idx_col == 0:
            ax.set_title(f"Modalidade {c}\n({len(idxs_cluster)} imagens)", fontsize=11, fontweight='bold', loc='left')

plt.suptitle("Categorização de Modalidades Visuais via Clustering (K-Means)", fontsize=14, y=0.99)
plt.tight_layout()

#Salvar
mosaic_path = OUTPUT_DIR / "modalidades_clusters.png"
plt.savefig(mosaic_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Mosaico ilustrativo salvo com sucesso em: {mosaic_path}")

# Split Estratificado (70, 15, 15)
train_samples, temp_samples, train_labels, temp_labels = train_test_split(
    sample_dirs, cluster_labels, test_size=0.30, stratify=cluster_labels, random_state=42
)

val_samples, test_samples, _, _ = train_test_split(
    temp_samples, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

print(f"\nResumo do Split Estratificado:")
print(f" - Treino:    {len(train_samples)} imagens (70%)")
print(f" - Validação: {len(val_samples)} imagens (15%)")
print(f" - Teste:     {len(test_samples)} imagens (15%)")

# Organiza os IDs/nomes das pastas para cada partição
split_data = {
    "train": [p.name for p in train_samples],
    "val": [p.name for p in val_samples],
    "test": [p.name for p in test_samples]
}

# Salva o mapeamento em JSON dentro de data/processed
split_json_path = OUTPUT_DIR / "split.json"
with open(split_json_path, "w") as f:
    json.dump(split_data, f, indent=4)

print(f"\nMapeamento dos splits salvo com sucesso em: {split_json_path}")