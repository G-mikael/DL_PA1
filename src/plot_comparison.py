import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuração de caminhos
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "results" / "logs"
OUTPUT_GRAPH = ROOT_DIR / "reports" / "figures" / "parte2_comparacao_baseline_vs_trilhaA.png"

# Carrega os resultados
def load_results(json_name):
    path = LOG_DIR / json_name
    if not path.exists():
        print(f"Erro: Arquivo {path} não encontrado.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

baseline_data = load_results("parte1_evaluation_metrics.json")
trilha_a_data = load_results("trilha_a_evaluation_metrics.json")

if baseline_data and trilha_a_data:
    # --- Extração das Métricas Gerais ---
    metrics_base = baseline_data["metrics"]
    metrics_ta = trilha_a_data["metrics"]
    
    # --- Extração dos Dados por Imagem ---
    # Ordenando pelo ID para garantir pareamento exato (caso a ordem mude)
    base_results = {r["id"]: r for r in baseline_data["per_image_results"]}
    ta_results = {r["id"]: r for r in trilha_a_data["per_image_results"]}
    
    common_ids = set(base_results.keys()).intersection(set(ta_results.keys()))
    
    gt_counts = [base_results[img_id]["gt_count"] for img_id in common_ids]
    maps_base = [base_results[img_id]["map"] for img_id in common_ids]
    maps_ta = [ta_results[img_id]["map"] for img_id in common_ids]

    # --- Criação da Figura ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Barras Comparativas de mAP
    labels = ['Baseline (Ingênua)', 'Trilha A (Watershed)']
    maps = [metrics_base["mAP_instance_050_095"], metrics_ta["mAP_instance_050_095"]]
    colors = ['#e63946', '#457b9d']
    
    bars = ax1.bar(labels, maps, color=colors, width=0.5)
    ax1.set_title("Comparação do mAP Global", fontsize=14, pad=15)
    ax1.set_ylabel("mAP (IoU 0.50:0.95)", fontsize=12)
    ax1.set_ylim(0, max(maps) + 0.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Adicionando os valores em cima das barras
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Gráfico 2: Dispersão mAP vs Densidade
    ax2.scatter(gt_counts, maps_base, alpha=0.6, color='#e63946', label='Baseline')
    ax2.scatter(gt_counts, maps_ta, alpha=0.6, color='#457b9d', marker='^', label='Trilha A')
    
    # Linhas de tendência (polinômio de grau 1)
    z_base = np.polyfit(gt_counts, maps_base, 1)
    p_base = np.poly1d(z_base)
    ax2.plot(sorted(gt_counts), p_base(sorted(gt_counts)), color='#e63946', linestyle='--', linewidth=2)
    
    z_ta = np.polyfit(gt_counts, maps_ta, 1)
    p_ta = np.poly1d(z_ta)
    ax2.plot(sorted(gt_counts), p_ta(sorted(gt_counts)), color='#457b9d', linestyle='-', linewidth=2)

    ax2.set_title("mAP vs Densidade de Objetos", fontsize=14, pad=15)
    ax2.set_xlabel("Número de Instâncias Reais na Imagem", fontsize=12)
    ax2.set_ylabel("mAP de Instância", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    
    # Salvamento
    OUTPUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_GRAPH, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo salvo com sucesso em: {OUTPUT_GRAPH}")