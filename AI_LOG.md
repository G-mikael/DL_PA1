# AI Log - Registro de Uso de Inteligência Artificial

As ferramentas de Inteligência Artificial foram utilizadas durante o desenvolvimento deste trabalho como apoio à programação, revisão de código, depuração e discussão de conceitos teóricos. As sugestões geradas pela IA foram revisadas e adaptadas à estrutura do projeto pela dupla.

Os trechos de código sugeridos pela IA foram revisados, adaptados e testados pela dupla. A IA foi utilizada como ferramenta de apoio, enquanto as decisões de implementação, os experimentos e a interpretação dos resultados foram realizadas e verificadas pela dupla.

Abaixo, registramos alguns episódios representativos do uso da IA durante o desenvolvimento.

## Episódio 1: Planejamento Computacional e Desenho das Ablações (Parte 3)

**O Problema:** Inicialmente, planejamos executar as ablações do Eixo 1 (Recuperação de Resolução) e do Eixo 2 (Função de Perda). Ao estimar o número de treinamentos necessário para as diferentes configurações e duas seeds por configuração, percebemos que o custo computacional seria elevado para o ambiente disponível.

**Como a IA ajudou:** Utilizamos a IA para analisar os tempos registrados nos experimentos anteriores e discutir alternativas para reduzir o custo experimental. A partir dos logs e do orçamento computacional disponível, decidimos utilizar 25 épocas nas ablações. A IA também auxiliou na organização dos scripts para que diferentes seeds, arquiteturas e checkpoints fossem salvos sem sobrescrever resultados anteriores.

## Episódio 2: Construção do Target de Três Classes (Parte 2)

**O Problema:** A baseline da Parte 1 utilizava uma máscara semântica binária, enquanto a Trilha A exigia uma representação com três classes: fundo, interior e fronteira.

**Como a IA ajudou:** Utilizamos a IA para discutir como transformar as máscaras individuais do DSB2018 em um target de três classes. A partir dessa discussão, implementamos a erosão morfológica de cada máscara para definir uma região de interior e utilizamos a diferença entre a máscara original e a região erodida para definir a fronteira. Também utilizamos a IA para verificar as dimensões esperadas das saídas da rede e do target para a utilização da Cross Entropy multiclasse.

## Episódio 3: Lógica de Fusão de Instâncias na Inferência em Mosaico (Parte 4)

**O Problema:** Na Parte 4, precisávamos lidar com instâncias que aparecem cortadas em mais de um tile durante a inferência em mosaico. A lógica para identificar fragmentos correspondentes na região de sobreposição e fundi-los era mais complexa do que o restante do pipeline.

**Como a IA ajudou:** Utilizamos a IA para esboçar a lógica da função de processamento do mosaico e discutir estratégias de comparação entre instâncias na região de overlap. O código sugerido foi posteriormente adaptado, testado e integrado ao restante do pipeline.

Além disso, a IA auxiliou na depuração de um erro no carregamento dos checkpoints, causado pela instanciação de uma arquitetura diferente daquela utilizada durante o treinamento, como no caso de pesos do DeepLabV3+ sendo carregados em uma U-Net.

## Episódio 4: Conexão entre Resultados Empíricos e Teoria (Parte 3)

**O Problema:** Após a ablação entre U-Net, DeepLabV3+ e PSPNet, precisávamos interpretar as diferenças observadas nas métricas e relacioná-las aos mecanismos de recuperação de resolução e de agregação de contexto apresentados em aula.

**Como a IA ajudou:** Utilizamos a IA como apoio para revisar a interpretação dos resultados, discutindo a função do pyramid pooling da PSPNet e a importância da informação espacial fina para a segmentação baseada em fronteiras seguida de Watershed. A partir dessa discussão, formulamos hipóteses para explicar o comportamento observado nas métricas e as confrontamos com os resultados experimentais.