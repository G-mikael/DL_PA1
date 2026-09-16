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
├── README.md
├── AI_LOG.md
├── requirements.txt
│
├── inferencia.ipynb
│
├── data/
│   ├── raw/                    # Dados originais
│   ├── processed/              # Dados processados
│   └── synthetic/              # Dados sintéticos
│
├── checkpoints/                # Pesos dos modelos treinados (.pth)
│
├── src/
│   ├── dataset.py              # Dataset, DataLoaders e augmentations
│   ├── synthetic.py            # Gerador de imagens sintéticas
│   ├── download_data.py        # Download/preparação dos dados
│   ├── prepare_dataset.py      # Preparação do dataset
│   ├── prepare_datasetpart6.py # Preparação para a Parte 6
│   ├── train_part6.py          # Treinamento da Parte 6
│   │
│   ├── models/
│   │   ├── fcn.py
│   │   ├── segnet.py
│   │   ├── unet.py
│   │   ├── resunet.py
│   │   ├── deeplab.py
│   │   └── pspnet.py
│   │
│   ├── losses/                 # Funções de perda
│   │
│   └── utils/
│       ├── matching.py         # Matching entre predições e ground truth
│       ├── map.py              # Métricas/mAP
│       ├── postprocessing.py   # Pós-processamento das predições
│       └── receptive_field.py  # Análise de campo receptivo
│
└── scripts/
    ├── train.py                # Treinamento
    ├── evaluate.py             # Avaliação
    └── stress_test.py          # Testes de estresse
```

---

## 3. Configuração do Ambiente

O projeto pode ser executado localmente ou em ambientes como **Google Colab**.

### 3.1. Clonar o repositório

```bash
git clone https://github.com/G-mikael/DL_PA1.git
cd DL_PA1
```

Caso esteja utilizando Google Colab:

```python
!git clone https://github.com/G-mikael/DL_PA1.git
%cd DL_PA1
```

### 3.2. Instalar as dependências

```bash
pip install -r requirements.txt
```

No Google Colab:

```python
!pip install -r requirements.txt
```

---

## 4. Dados

O projeto utiliza imagens contendo múltiplos objetos, acompanhadas de suas respectivas anotações de instância.

O diretório `data/` deve conter os dados necessários para treinamento e avaliação.

A estrutura é:

```text
data/
├── raw/
├── processed/
└── synthetic/
```


### Dados sintéticos

O projeto possui um gerador de imagens sintéticas baseado em **elipses**, permitindo criar imagens contendo múltiplas instâncias com diferentes posições, tamanhos e configurações.

A geração dos dados é realizada por:

```bash
python src/synthetic.py
```

---

## 5. Preparação dos Dados

Para realizar o download e a preparação inicial dos dados:

```bash
python src/download_data.py
python src/prepare_dataset.py
```

No Google Colab:

```python
!python src/download_data.py
!python src/prepare_dataset.py
```

Para a preparação específica da **Parte 6**, utilizando validação cruzada **Leave-One-Cluster-Out**:

```bash
python src/prepare_datasetpart6.py
```

No Google Colab:

```python
!python src/prepare_datasetpart6.py
```

---

## 6. Treinamento

O treinamento dos modelos pode ser realizado através dos scripts disponíveis em `src/` e `scripts/`.

Para o treinamento utilizado na Parte 2:

```bash
python src/train_part2_s2.py
```

No Google Colab:

```python
!python src/train_part2_s2.py
```

Os pesos dos modelos treinados são armazenados em:

```text
checkpoints/
```

com extensão `.pth`.

Dependendo da configuração utilizada, o treinamento pode utilizar GPU CUDA quando disponível.

---

## 7. Avaliação

A avaliação dos modelos pode ser executada através do script:

```bash
python scripts/evaluate_models.py
```

No Google Colab:

```python
!python scripts/evaluate_models.py
```

A avaliação considera as métricas implementadas no projeto, incluindo o processo de matching entre as instâncias preditas e as instâncias presentes no ground truth.

Os procedimentos de avaliação também podem ser utilizados para comparar diferentes arquiteturas e configurações de treinamento.

---

## 8. Inferência

Para realizar inferência sobre imagens individuais, utilize o notebook:

```text
script/inferencia.ipynb
```

O notebook permite carregar um checkpoint treinado, fornecer uma imagem de entrada e visualizar as instâncias segmentadas pelo modelo.

Os pesos utilizados na inferência devem estar disponíveis no diretório:

```text
checkpoints/
```

---

## 9. Arquiteturas

O projeto investiga a adaptação de diferentes arquiteturas de segmentação semântica para segmentação de instâncias.

Entre as arquiteturas consideradas estão:

* **U-Net**
* **ResUNet**
* **DeepLab**
* **PSPNet**

A principal diferença em relação à segmentação semântica convencional está na representação utilizada para distinguir objetos diferentes pertencentes à mesma classe.

---

## 10. Reprodutibilidade

Para reproduzir os experimentos, recomenda-se:

1. Clonar o repositório.
2. Instalar as dependências especificadas em `requirements.txt`.
3. Preparar os dados.
4. Executar a preparação específica da etapa desejada.
5. Treinar o modelo utilizando o script correspondente.
6. Avaliar o modelo.
7. Utilizar `inferencia.ipynb` para visualizar os resultados.

Os checkpoints gerados durante o treinamento devem ser mantidos em `checkpoints/`.

---

## 11. Registro do Uso de Inteligência Artificial

O arquivo `AI_LOG.md` contém o registro das ferramentas de inteligência artificial generativa utilizadas durante o desenvolvimento do projeto, conforme solicitado pela disciplina.

---

## 18. Autores

**Gerardo Mikael do Carmo Pereira**
**George Rodrigues Vaz**

Disciplina de Aprendizado Profundo.
