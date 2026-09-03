# PA1 - Segmentação de Instâncias sem Detectores de Região

**Disciplina:** Aprendizado Profundo  
**Professor:** Dario Oliveira | **Monitor:** Erick Brito  
**Integrantes:** Gerardo Mikael do Carmo Pereira & George Rodrigues Vaz

---

## 1. Descrição do Projeto

Este repositório contém a implementação do **Programming Assignment 1 (PA1)**. O objetivo do projeto é adaptar arquiteturas clássicas de segmentação semântica (FCN, SegNet, U-Net, ResUNet, DeepLab, PSPNet) para a tarefa de **segmentação de instâncias**, sem o uso de redes com proposta de região (ex: Mask R-CNN) ou modelos pré-treinados específicos para instâncias.

A solução envolve o projeto de três componentes interligados:
1. **Representação de saída:** O que a rede prevê por pixel ou área.
2. **Função de Perda (Loss):** Otimização adequada ao desbalanceamento de classes e regressão de coordenadas/distâncias.
3. **Decodificação / Pós-processamento:** Extração de instâncias individuais a partir dos mapas previstos.

---

## 2. Estrutura do Repositório

```text
.
├── README.md                 # Guia do projeto e instruções de execução
├── AI_LOG.md                 # Registro do uso de ferramentas de IA generativa
├── requirements.txt          # Dependências do ambiente Python
├── inferencia.ipynb          # Notebook para inferência em imagem individual
├── data/                     # Diretório de dados com o dataset bruto e o sintético
├── checkpoints/              # Pesos dos modelos treinados (.pth)
├── src/                      # Código-fonte modularizado
│   ├── dataset.py            # Datasets, DataLoaders e Augmentations
│   ├── synthetic.py          # Gerador sintético de elipses
│   ├── models/               # Arquiteturas de rede
│   ├── losses/               # Funções de perda personalizadas
│   └── utils/                # Matching, mAP, pós-processamento e campo receptivo
└── scripts/                  # Scripts de entrada
    ├── train.py              # Script principal de treinamento
    ├── evaluate.py           # Script de avaliação e cálculo de métricas
    └── stress_test.py        # Execução dos testes de estresse