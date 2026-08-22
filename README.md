# MS-PESO

Estimativa não invasiva do peso vivo de bovinos por visão computacional.

O primeiro marco do projeto é um baseline reproduzível que recebe uma imagem
RGB lateral de um bovino e retorna o peso estimado em quilogramas. O projeto é
pensado para evoluir para segmentação, múltiplas vistas, profundidade e uma
aplicação de campo, sem misturar essas etapas antes de termos uma avaliação
confiável.

## Decisão central

O MVP considera um cenário controlado:

- um animal por imagem;
- câmera e local de captura conhecidos;
- vista lateral com o corpo inteiro visível;
- peso de balança coletado no mesmo evento da imagem;
- treino, validação e teste separados por `animal_id`.

Uma foto monocular sem referência de escala não determina tamanho físico de
forma inequívoca. Por isso, o protocolo recomenda câmera fixa e um marcador de
dimensão conhecida. Fotografias livres de celular serão uma etapa posterior.

## Regra principal de engenharia: SRP

Este projeto deve seguir obrigatoriamente o **Single Responsibility Principle
(SRP)**. Cada módulo, classe ou função deve ter uma responsabilidade coesa e um
único motivo principal para mudar.

- aquisição, validação, transformação, treinamento, avaliação e inferência
  devem permanecer em componentes separados;
- módulos de orquestração podem coordenar componentes, mas não devem absorver
  suas regras internas;
- uma alteração que introduza uma segunda responsabilidade deve ser refatorada
  antes de ser aceita;
- conveniência ou velocidade de prototipação não justificam concentrar todo o
  pipeline em um notebook ou arquivo monolítico.

Os critérios completos estão em
[Princípios de engenharia](docs/ENGINEERING_PRINCIPLES.md).

## Estado atual

- [x] Escopo e critérios do MVP documentados.
- [x] Contrato do manifesto de dados definido.
- [x] Protocolo inicial de coleta definido.
- [x] Baseline ResNet18 para regressão estruturado.
- [x] Divisão agrupada por animal implementada.
- [x] Estrutura e metadados do CowDB auditados.
- [x] ZIP público longitudinal de Nelore/UFGD auditado sem download integral.
- [x] Inventariador SRP para o dataset UAV de Nelore implementado.
- [x] CowDB baixado e manifesto lateral validado com 154 animais.
- [x] Baseline B0 da média executado sobre o conjunto de teste.
- [x] ResNet18 e EfficientNet-B0 treinadas em GPU sobre a mesma divisão CowDB.
- [x] EfficientNet-B0 balanceada promoveu o baseline visual para MAE de 32,55 kg.
- [ ] Coleta piloto de bovinos-alvo iniciada.

## Estrutura

```text
configs/                    configurações dos experimentos
data/                       contrato e exemplos de dados
docs/                       plano, protocolo e experimentos
src/ms_peso/                pacote Python
tests/                      testes que evitam vazamento entre animais
artifacts/                  métricas e modelos gerados (não versionados)
```

## Instalação

Recomendado: Python 3.10 ou 3.11, com GPU no Google Colab para treinamento.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

O ambiente local atual ainda não possui PyTorch funcional; portanto o primeiro
treinamento deverá ser feito após a instalação acima ou no Colab.

## Preparar o manifesto

Obtenha uma cópia da fonte oficial diretamente no diretório ignorado pelo Git:

```bash
git clone --depth 1 https://github.com/ruchaya/CowDB.git data/raw/cowdb
```

Depois, converta o CowDB para o contrato interno:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_manifest.csv \
  --views left
```

Para dados próprios, crie um CSV seguindo [data/README.md](data/README.md). Se o
manifesto ainda não possuir a coluna `split`, gere uma divisão agrupada por
bovino:

```bash
python -m ms_peso.prepare_manifest \
  --input data/interim/manifest.csv \
  --output data/processed/manifest.csv \
  --image-root data \
  --seed 42
```

O comando falha se o mesmo `animal_id` aparecer em mais de uma partição.

O baseline ingênuo obrigatório pode ser reproduzido sem GPU:

```bash
python -m ms_peso.evaluate_mean_baseline \
  --manifest data/processed/manifest.csv \
  --image-root data \
  --view left \
  --check-images
```

Na divisão atual do CowDB, ele obteve MAE de 53,05 kg, RMSE de 74,81 kg e MAPE
de 13,83%. O primeiro modelo visual deve superar esse resultado no mesmo teste.

O ZIP público do NelloreBeefCattleDataset foi classificado como fonte auxiliar:
ele contém imagens UAV e anotações de detecção/segmentação, mas não contém
pesos nem identificação persistente dos animais. Depois de extrair o arquivo em
um volume com espaço suficiente, gere um inventário reproduzível com:

```bash
python -m ms_peso.inspect_nellore_uav \
  --dataset-root /caminho/NelloreBeefCattleDataset \
  --output artifacts/nellore_uav_inventory.json
```

Esse comando audita a fonte; deliberadamente não cria um manifesto de regressão.

## Treinar o baseline

```bash
python -m ms_peso.train --config configs/baseline_rgb.yaml
```

O treinamento grava em `artifacts/baseline_rgb/`:

- `best_model.pt`: pesos e normalização do alvo;
- `metrics.json`: MAE, RMSE, MAPE e R²;
- `predictions_test.csv`: peso real e estimado por amostra;
- `resolved_manifest.csv`: divisão efetivamente usada.

O baseline EfficientNet-B0 com amostragem uniforme é reproduzido com:

```bash
python -m ms_peso.train --config configs/efficientnet_b0_rgb.yaml
```

A variante com amostragem moderada por faixa reduziu o MAE para 32,55 kg:

```bash
python -m ms_peso.train --config configs/efficientnet_b0_balanced.yaml
```

## Documentos do projeto

- [Plano do produto e pesquisa](docs/PROJECT_PLAN.md)
- [Protocolo de coleta](docs/DATA_COLLECTION_PROTOCOL.md)
- [Plano de experimentos](docs/EXPERIMENTS.md)
- [Registro de decisões](docs/DECISIONS.md)
- [Execução no Google Colab](docs/COLAB.md)
- [Princípios de engenharia](docs/ENGINEERING_PRINCIPLES.md)
- [Datasets e auditoria](docs/DATASETS.md)
- [Fontes brasileiras de dados](docs/BRAZILIAN_DATA_SOURCES.md)

## Fontes públicas iniciais

Para o domínio final, priorizamos fontes brasileiras de Nelore:

- [NelloreBeefCattleDataset](https://github.com/EvertonTetila/NelloreBeefCattleDataset):
  o ZIP longitudinal público tem 904 imagens dorsais por UAV e rótulos de
  detecção/segmentação, sem peso ou `animal_id` persistente;
- [UNESP — imagens 3D e peso](https://pmc.ncbi.nlm.nih.gov/articles/PMC10215216/):
  450 animais Nelore; dados mediante solicitação, que será feita posteriormente;
- [Nelore Instance Segmentation](https://universe.roboflow.com/henriques-workspace-lhcrk/nelore-instance-segmentation):
  fonte auxiliar para segmentação, sem peso identificado.

Consulte [Fontes brasileiras de dados](docs/BRAZILIAN_DATA_SOURCES.md) para as
diferenças entre versões, licenças e usos permitidos.

Como baseline de engenharia internacional:

- [CowDB](https://github.com/ruchaya/CowDB): 154 bovinos Hereford, RGB-D,
  múltiplas vistas, peso e medidas corporais.
- [CowDatabase2](https://github.com/ruchaya/CowDatabase2): 119 bovinos Black
  Angus, RGB-D, múltiplas vistas, peso e medidas corporais.

Essas bases servem para validar o pipeline. Um modelo destinado a Nelore ou a
condições brasileiras precisará ser ajustado e testado em dados locais.

O CowDB não inclui uma licença explícita na revisão auditada. Consulte
[Datasets e auditoria](docs/DATASETS.md) antes de redistribuir dados ou derivados.
