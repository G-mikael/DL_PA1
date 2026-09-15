import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "results" / "logs"

def get_map(json_name):
    try:
        with open(LOG_DIR / json_name, "r") as f:
            return json.load(f)["metrics"]["mAP"]
    except FileNotFoundError:
        return 0

# Agrupando as seeds
map_unet = [get_map("eval_unet_2027.json"), get_map("eval_unet_2028.json")]
map_deeplab = [get_map("eval_deeplab_2027.json"), get_map("eval_deeplab_2028.json")]
map_pspnet = [get_map("eval_pspnet_2027.json"), get_map("eval_pspnet_2028.json")]

# Cálculo de Média e Desvio Padrão
means = [np.mean(map_unet), np.mean(map_deeplab), np.mean(map_pspnet)]
stds = [np.std(map_unet), np.std(map_deeplab), np.std(map_pspnet)]
labels = ['U-Net\n(Skip Connections)', 'DeepLab\n(Atrous/ASPP)', 'PSPNet\n(Contexto Global)']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, means, yerr=stds, capsize=10, color=['#e63946', '#457b9d', '#2a9d8f'], alpha=0.8)

ax.set_title("Comparação de Arquiteturas - Ablação (Parte 3)", fontsize=14, pad=15)
ax.set_ylabel("mAP de Instância", fontsize=12)
ax.set_ylim(0, max(means) + 0.1)

for bar, std in zip(bars, stds):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + std + 0.01, f"{yval:.4f} ±{std:.4f}", ha='center', va='bottom', fontweight='bold')

output_path = LOG_DIR.parent.parent / "reports" / "figures" / "ablations_comparison.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("Gráfico de ablação gerado com sucesso!")