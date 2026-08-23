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
- [x] Literatura UAV brasileira revisada: fonte 3D de 2026 priorizada e
  inconsistência da disponibilidade anunciada em 2025 documentada.
- [x] CowDB baixado e manifesto lateral validado com 154 animais.
- [x] Baseline B0 da média executado sobre o conjunto de teste.
- [x] ResNet18 e EfficientNet-B0 treinadas em GPU sobre a mesma divisão CowDB.
- [x] EfficientNet-B0 atingiu MAPE de 9,39% e permanece a referência visual.
- [x] Variante balanceada atingiu MAE pontual de 32,55 kg, sem superioridade
  estatística confirmada no teste atual.
- [x] Três seeds confirmaram o B2 uniforme como mais estável: MAE médio de
  32,10 ± 2,43 kg contra 36,03 ± 6,68 kg da variante balanceada.
- [x] ConvNeXt-Tiny testada em três seeds; não promovida por piorar RMSE,
  MAPE, viés e R² apesar de MAE médio 0,46 kg menor.
- [x] Fusão RGB + profundidade testada; não promovida após piorar o MAE em
  9,06 kg no bootstrap pareado contra o B2.
- [x] Recorte e máscara guiados por profundidade avaliados; a máscara chegou a
  MAE de 28,16 kg, mas foi rejeitada após variar até 45,00 kg entre seeds.
- [x] Fusão lateral + superior avaliada; não promovida por piorar MAE, MAPE e
  principalmente o viés em relação ao B2 na comparação pareada.
- [x] Altura física extraída das nuvens PLY e auditada somente em
  treino/validação; não fundida ao B2 por não explicar seus resíduos.
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

O ambiente local validado usa Python 3.12, PyTorch 2.12.1 com CUDA 13.0 e uma
NVIDIA GeForce RTX 3060 Ti. A instalação também pode ser feita no Colab.

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

A arquitetura ConvNeXt-Tiny também foi avaliada, mas não substituiu o B2:

```bash
python -m ms_peso.train --config configs/convnext_tiny_rgb.yaml
```

O carregador experimental RGB + profundidade pode ser reproduzido com
`configs/efficientnet_b0_rgb_depth.yaml`. A fusão da cena bruta não superou o
B2; recorte e máscara foram avaliados separadamente na etapa seguinte.

O preparador `python -m ms_peso.prepare_depth_crops` estima o fundo somente com
o treino e gera recortes ou canvases mascarados com trava de qualidade. Nem o
recorte nem a máscara superaram a estabilidade do B2 em três seeds.

O experimento multivista combina a lateral e a superior do mesmo evento em uma
única amostra. As duas vistas passam por um EfficientNet-B0 compartilhado:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_left_top_rows.csv \
  --views left top
python -m ms_peso.prepare_multi_view_manifest \
  --input data/interim/cowdb_left_top_rows.csv \
  --output data/interim/cowdb_left_top_paired.csv \
  --image-root data
python -m ms_peso.prepare_manifest \
  --input data/interim/cowdb_left_top_paired.csv \
  --output data/processed/left_top_manifest.csv \
  --image-root data \
  --check-images \
  --seed 42
python -m ms_peso.train --config configs/efficientnet_b0_left_top.yaml
```

Essa configuração foi preservada para reprodução, mas não substituiu o B2.

As nuvens PLY organizadas do CowDB podem ser auditadas sem consultar o teste.
O comando abaixo mede uma altura robusta após subtração do fundo calculado
somente no treino e compara seu sinal com os resíduos de validação do B2:

```bash
python -m ms_peso.audit_point_cloud_geometry \
  --manifest data/processed/rgb_depth_point_cloud_manifest.csv \
  --image-root data \
  --output artifacts/point_cloud_geometry_audit/report.json \
  --reference-checkpoint artifacts/efficientnet_b0_rgb/best_model.pt
```

Esse gate rejeitou a fusão antes do treinamento: a correlação entre altura e
resíduo do B2 na validação foi -0,016, praticamente nula. As vistas direita e
superior também foram auditadas; seus valores foram -0,106 e +0,210. O
argumento opcional `--reference-manifest` permite alinhar a geometria de outra
vista ao RGB lateral pelo mesmo animal/evento.

A variante experimental com amostragem moderada por faixa obteve MAE pontual
de 32,55 kg:

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
- [Embrapa/UFGD — nuvem de pontos UAV e peso](https://www.alice.cnptia.embrapa.br/alice/handle/doc/1187153):
  estudo de 2026 com pesagem no dia do voo e RMSE relatado de 8,35 kg; dados e
  tamanho final da avaliação precisam ser solicitados;
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
