# Plano de experimentos

## Regras comuns

Todos os experimentos usam a mesma divisão por animal e registram seed,
manifesto, arquitetura, pesos pré-treinados, transformações, hiperparâmetros e
métricas. Uma mudança por vez sempre que o objetivo for comparação.

## Baselines

| ID | Entrada | Modelo | Objetivo |
|---|---|---|---|
| B0 | nenhuma imagem | média do treino | limite ingênuo obrigatório |
| B1 | RGB lateral | ResNet18 | primeiro baseline treinável |
| B2 | RGB lateral | EfficientNet-B0 | arquitetura eficiente |
| B3 | RGB lateral | ConvNeXt-Tiny | backbone moderno |

## Execuções registradas

### B0-CowDB-001 — média do treino

- data: 2026-08-22;
- fonte: CowDB, commit `270f2908de9a6931789400fbea122fe5e8df35b6`;
- entrada: vista lateral esquerda;
- divisão: seed 42, com 109/25/20 animais em treino/validação/teste;
- SHA-256 do manifesto:
  `b623d4a54a8d4378a576e8611258e3fb9efecb52f2c3bc7c0310268db1ad122d`;
- predição constante: média do treino de 452,2477 kg;
- teste: 20 animais nunca vistos.

| Métrica | Resultado |
|---|---:|
| MAE | 53,05 kg |
| RMSE | 74,81 kg |
| MAPE | 13,83% |
| Viés médio | +1,45 kg |
| R² | -0,0004 |
| dentro de ±10 kg | 10% |
| dentro de ±20 kg | 35% |

Artefatos locais: `artifacts/baseline_mean/metrics.json` e
`artifacts/baseline_mean/predictions_test.csv`. Eles são ignorados pelo Git e
devem ser preservados junto da execução.

### B1-CowDB-001 — ResNet18 RGB lateral

- data: 2026-08-22;
- entrada: vista lateral esquerda, 224 × 224 pixels;
- divisão: a mesma de B0, com 109/25/20 animais;
- inicialização: pesos ImageNet;
- treinamento: Huber Loss, AdamW, seed 42 e parada antecipada;
- melhor época de validação: 20; encerramento na época 27;
- dispositivo: NVIDIA GeForce RTX 3060 Ti, PyTorch 2.12.1 + CUDA 13.0.

| Métrica | B1 | B0 | Variação de B1 |
|---|---:|---:|---:|
| MAE | 37,55 kg | 53,05 kg | -29,2% |
| RMSE | 60,97 kg | 74,81 kg | -18,5% |
| MAPE | 10,77% | 13,83% | -22,1% |
| Viés médio | +21,65 kg | +1,45 kg | pior |
| R² | 0,336 | -0,0004 | melhor |

O B1 supera o baseline ingênuo, mas ainda não atinge o MAPE piloto abaixo de
10% e apresenta viés positivo relevante. A análise por faixa e os maiores
erros são gerados em `artifacts/baseline_rgb/error_analysis.md` pelo comando:

```bash
python -m ms_peso.error_analysis \
  --predictions artifacts/baseline_rgb/predictions_test.csv \
  --output-dir artifacts/baseline_rgb \
  --label B1-CowDB-001
```

A inspeção dos dois maiores erros não indicou rótulos incorretos ou falha grave
de enquadramento. O conjunto de treino contém apenas quatro animais abaixo de
350 kg; os erros extremos são compatíveis com regressão à média nessa faixa
rara. O experimento B2 mantém a mesma divisão e hiperparâmetros para isolar a
troca de arquitetura.

### B2-CowDB-001 — EfficientNet-B0 RGB lateral

- data: 2026-08-22;
- entrada, divisão e hiperparâmetros: iguais ao B1;
- melhor época de validação: 8; encerramento na época 15;
- dispositivo: NVIDIA GeForce RTX 3060 Ti.

| Métrica | B2 | B1 | Variação de B2 |
|---|---:|---:|---:|
| MAE | 34,21 kg | 37,55 kg | -8,9% |
| RMSE | 51,06 kg | 60,97 kg | -16,3% |
| MAPE | 9,39% | 10,77% | -12,8% |
| Viés médio | +7,07 kg | +21,65 kg | melhor |
| R² | 0,534 | 0,336 | melhor |

O B2 supera o B1 e atinge o critério piloto de MAPE abaixo de 10%. Ele reduz a
superestimação dos dois animais abaixo de 350 kg, mas ainda apresenta MAE de
133,14 kg nessa faixa rara e subestima em média 41,85 kg os quatro animais com
500 kg ou mais. B2 passa a ser o melhor baseline visual, sem evidência ainda
suficiente para uso em campo.

### A7-CowDB-001 — amostragem balanceada moderada

- data: 2026-08-22;
- modelo: EfficientNet-B0 e mesma divisão de B2;
- mudança isolada: amostragem com reposição pela raiz quadrada do inverso da
  frequência nas faixas `<350`, `350–399`, `400–449`, `450–499` e `≥500 kg`;
- melhor época de validação: 23; treinamento encerrado na época 30.

| Métrica | A7 | B2 | Variação de A7 |
|---|---:|---:|---:|
| MAE | 32,55 kg | 34,21 kg | -4,9% |
| RMSE | 50,60 kg | 51,06 kg | -0,9% |
| MAPE | 9,12% | 9,39% | -2,9% |
| Viés médio | +12,72 kg | +7,07 kg | pior |
| R² | 0,542 | 0,534 | melhor |

Nos 18 animais com pelo menos 350 kg, o A7 obtém MAE de 21,20 kg e viés de
-0,84 kg. Na faixa abaixo de 350 kg, o MAE permanece muito alto (134,76 kg),
praticamente igual ao B2; não há dados suficientes para resolver essa faixa
apenas por reamostragem.

### Comparação pareada B2 × A7

Foram executadas 10.000 reamostragens bootstrap pareadas dos mesmos 20 animais,
com seed 42. A diferença de MAE de A7 menos B2 foi de -1,66 kg, com IC95% de
-6,62 a +3,06 kg e probabilidade bootstrap de A7 ter MAE menor de 74,2%. Para
MAPE, a diferença foi de -0,27 ponto percentual, com IC95% de -1,28 a +0,71.

Como os intervalos pareados cruzam zero, a vantagem pontual do A7 não está
estatisticamente sustentada. O B2 permanece a referência principal por ser mais
simples e apresentar menor viés global. A avaliação seguinte verifica a
estabilidade dessa conclusão em outras seeds.

A comparação pode ser reproduzida com:

```bash
python -m ms_peso.compare_models \
  --reference artifacts/efficientnet_b0_rgb/predictions_test.csv \
  --candidate artifacts/efficientnet_b0_balanced/predictions_test.csv \
  --reference-label B2-CowDB-001 \
  --candidate-label A7-CowDB-001 \
  --output-dir artifacts/comparison_b2_a7 \
  --iterations 10000 \
  --seed 42
```

### Estabilidade em três seeds

B2 e A7 foram repetidos com seeds 42, 43 e 44, sem alterar a divisão dos
animais. Os resultados abaixo mostram média e desvio-padrão amostral:

| Métrica | B2 uniforme | A7 balanceado | Diferença A7 − B2 |
|---|---:|---:|---:|
| MAE | 32,10 ± 2,43 kg | 36,03 ± 6,68 kg | +3,92 kg |
| RMSE | 49,11 ± 4,36 kg | 53,96 ± 6,61 kg | +4,85 kg |
| MAPE | 8,84 ± 0,73% | 9,63 ± 1,12% | +0,79 p.p. |
| Viés | +7,05 ± 2,29 kg | +5,69 ± 11,72 kg | -1,36 kg |
| R² | 0,57 ± 0,07 | 0,47 ± 0,13 | -0,09 |

A seed 43 do B2 produziu o melhor resultado isolado (MAE de 29,44 kg), mas a
comparação usa a média das três execuções. O A7 piora o erro médio e apresenta
instabilidade substancialmente maior; portanto ele não é promovido. O B2 com
amostragem uniforme permanece o baseline visual oficial.

O resumo pode ser reproduzido com `python -m ms_peso.summarize_runs`, passando
os três arquivos `metrics.json` de cada estratégia.

### B3-CowDB — ConvNeXt-Tiny RGB lateral

- data: 2026-08-22;
- entrada, divisão, aumentos e hiperparâmetros: iguais ao B2;
- mudança isolada: arquitetura ConvNeXt-Tiny com pesos ImageNet;
- seed 42: melhor época de validação 8; encerramento na época 15.

Na seed 42, o B3 obteve MAE de 32,42 kg, RMSE de 55,90 kg, MAPE de 9,42%,
viés de +11,87 kg e R² de 0,441. O MAE pontual foi 1,80 kg menor que o B2,
mas o bootstrap pareado de 10.000 iterações produziu IC95% de -8,92 a
+5,48 kg. A melhora não é sustentada; RMSE, MAPE e viés foram piores.

Os dois animais abaixo de 350 kg concentraram os maiores erros do B3, com MAE
de 151,27 kg nessa faixa. Nos outros 18 animais, o MAE foi 19,21 kg e o viés
-3,62 kg. Isso reforça que a aparente melhora global não resolve a região rara
e ainda amplia os erros extremos.

O experimento foi repetido com seeds 42, 43 e 44:

| Métrica | B2 EfficientNet-B0 | B3 ConvNeXt-Tiny | Diferença B3 − B2 |
|---|---:|---:|---:|
| MAE | 32,10 ± 2,43 kg | 31,64 ± 2,84 kg | -0,46 kg |
| RMSE | 49,11 ± 4,36 kg | 55,36 ± 6,07 kg | +6,25 kg |
| MAPE | 8,84 ± 0,73% | 9,28 ± 0,96% | +0,44 p.p. |
| Viés | +7,05 ± 2,29 kg | +14,05 ± 2,14 kg | +7,00 kg |
| R² | 0,57 ± 0,07 | 0,45 ± 0,12 | -0,12 |

O ganho muito pequeno de MAE não compensa a piora consistente das demais
métricas e dos erros grandes. O B3 não é promovido; B2 permanece o baseline
visual oficial.

Reprodução da execução principal:

```bash
python -m ms_peso.train --config configs/convnext_tiny_rgb.yaml
```

### A6-CowDB-001 — fusão RGB + profundidade

- data: 2026-08-22;
- entrada: RGB lateral e profundidade de 16 bits sincronizados pelo timestamp;
- divisão: a mesma de B2, com 109/25/20 animais;
- RGB: EfficientNet-B0 com pesos ImageNet;
- profundidade: codificador convolucional pequeno treinado do zero;
- fusão: características globais antes da regressão, sem presumir alinhamento
  pixel a pixel entre RGB 1920 × 1080 e profundidade 512 × 424;
- normalização da profundidade: faixa física fixa de 0 a 8.000 mm;
- melhor época de validação: 6; encerramento na época 13.

| Métrica | A6 | B2 | Variação de A6 |
|---|---:|---:|---:|
| MAE | 43,28 kg | 34,21 kg | +26,5% |
| RMSE | 59,56 kg | 51,06 kg | +16,6% |
| MAPE | 11,33% | 9,39% | +20,6% |
| Viés médio | -3,77 kg | +7,07 kg | menor magnitude |
| R² | 0,366 | 0,534 | pior |

O bootstrap pareado de 10.000 iterações confirmou a piora: a diferença A6
menos B2 foi +9,06 kg de MAE, IC95% de +1,31 a +16,81 kg; para RMSE, +8,50
kg, IC95% de +2,90 a +15,76 kg. Como os intervalos não cruzam zero, esta
configuração não justifica repetição em outras seeds.

Os dois animais abaixo de 350 kg continuaram concentrando os maiores erros,
com MAE de 144,60 kg. Nos 18 animais restantes, o A6 teve MAE de 32,02 kg,
também pior que os 19,21 kg do B3. A profundidade bruta contém a silhueta do
animal, porém inclui piso, cercas e mediana de aproximadamente 43% de pixels
inválidos. Estatísticas globais de profundidade apresentaram correlação fraca
com peso no conjunto completo.

**Decisão:** rejeitar fusão bruta da cena. A próxima tentativa com profundidade
deve usá-la para recorte/segmentação do bovino ou extrair geometria após remover
o fundo, mantendo essa transformação separada do regressor.

Reprodução:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_rgb_depth_manifest.csv \
  --views left \
  --include-depth
python -m ms_peso.prepare_manifest \
  --input data/interim/cowdb_rgb_depth_manifest.csv \
  --output data/processed/rgb_depth_manifest.csv \
  --image-root data \
  --seed 42
python -m ms_peso.train --config configs/efficientnet_b0_rgb_depth.yaml
```

### A1-CowDB-001 — recorte guiado por profundidade

- fundo de profundidade estimado exclusivamente com os 109 animais de treino;
- valor máximo observado por pixel usado como aproximação do fundo estático;
- primeiro plano definido por diferença mínima de 150 mm;
- caixa do maior componente com margem de 8%;
- trava de qualidade: caixa deve ocupar ao menos 50% da imagem;
- RGB recortado e redimensionado para 224 × 224;
- modelo, divisão, seed e hiperparâmetros iguais ao B2.

O primeiro protótipo com percentil 90 gerou três recortes parciais e foi
rejeitado antes do treinamento. Com o máximo por pixel, os 154 animais passaram
pela trava; as caixas ocuparam de 58,8% a 78,8% da cena e os casos extremos
foram inspecionados visualmente.

| Métrica | A1 | B2 | Diferença A1 − B2 |
|---|---:|---:|---:|
| MAE | 37,37 kg | 34,21 kg | +3,15 kg |
| RMSE | 53,58 kg | 51,06 kg | +2,52 kg |
| MAPE | 10,00% | 9,39% | +0,61 p.p. |
| Viés | +0,67 kg | +7,07 kg | -6,39 kg |
| R² | 0,487 | 0,534 | pior |

Os intervalos pareados de erro cruzaram zero, mas a probabilidade bootstrap de
A1 ter MAE menor foi somente 11,9%. O viés melhorou de forma sustentada. O
recorte não foi repetido em outras seeds porque piorou todas as métricas de erro
e altera a escala aparente ao ampliar caixas diferentes para o mesmo tamanho.

### A2-CowDB — máscara retangular sem alterar escala

O A2 reutiliza exatamente as caixas validadas do A1, mas mantém o canvas RGB
original. Somente a região externa à caixa é neutralizada com a média ImageNet;
assim o tamanho aparente do animal em relação à câmera é preservado.

Na seed 42, o A2 produziu o melhor resultado pontual do projeto:

| Métrica | A2 seed 42 | B2 seed 42 |
|---|---:|---:|
| MAE | 28,16 kg | 34,21 kg |
| RMSE | 42,37 kg | 51,06 kg |
| MAPE | 7,63% | 9,39% |
| Viés | +1,28 kg | +7,07 kg |
| R² | 0,679 | 0,534 |

No bootstrap pareado, A2 menos B2 teve diferença de MAE de -6,05 kg, IC95%
de -13,06 a +0,88 kg, e 95,7% de probabilidade de MAE menor. O intervalo ainda
cruza zero. O MAE dos dois animais abaixo de 350 kg caiu para 106,77 kg, uma
melhora importante, porém ainda inadequada.

As repetições revelaram que o ganho não é estável:

| Métrica | B2 uniforme | A2 máscara | Diferença A2 − B2 |
|---|---:|---:|---:|
| MAE | 32,10 ± 2,43 kg | 36,97 ± 8,44 kg | +4,86 kg |
| RMSE | 49,11 ± 4,36 kg | 51,65 ± 9,23 kg | +2,54 kg |
| MAPE | 8,84 ± 0,73% | 9,74 ± 1,96% | +0,90 p.p. |
| Viés | +7,05 ± 2,29 kg | -0,44 ± 6,07 kg | -7,48 kg |
| R² | 0,57 ± 0,07 | 0,51 ± 0,17 | -0,05 |

As seeds 42, 43 e 44 do A2 tiveram MAE de 28,16, 37,74 e 45,00 kg. A máscara
quase zera o viés médio, mas aumenta muito a variabilidade e piora o erro médio.
Ela não é promovida; B2 continua como baseline oficial.

Reprodução dos dados derivados e dos dois experimentos:

```bash
python -m ms_peso.prepare_depth_crops \
  --manifest data/processed/rgb_depth_manifest.csv \
  --image-root data \
  --output-dir data/interim/cowdb_depth_crops \
  --output-manifest data/processed/depth_crop_manifest.csv
python -m ms_peso.prepare_depth_crops \
  --manifest data/processed/rgb_depth_manifest.csv \
  --image-root data \
  --output-dir data/interim/cowdb_depth_box_masks \
  --output-manifest data/processed/depth_box_mask_manifest.csv \
  --output-mode masked_canvas
python -m ms_peso.train --config configs/efficientnet_b0_depth_crop.yaml
python -m ms_peso.train --config configs/efficientnet_b0_depth_box_mask.yaml
```

### A5-CowDB-001 — fusão lateral + superior

- entrada: uma vista lateral esquerda e uma superior do mesmo animal/evento;
- manifesto: 154 pares completos, SHA-256
  `cac0b63ba20aa3853f5c43c679b7bca92761780118905ba5386c8d27392a65f1`;
- divisão: os mesmos 109/25/20 animais do B2, sem duplicar animais como
  amostras independentes;
- modelo: um único EfficientNet-B0 compartilhado pelas duas vistas, com fusão
  dos dois vetores globais antes da regressão;
- seed 42: melhor época de validação 12; encerramento na época 19.

| Métrica | A5 | B2 | Diferença A5 − B2 |
|---|---:|---:|---:|
| MAE | 37,91 kg | 34,21 kg | +3,70 kg |
| RMSE | 53,09 kg | 51,06 kg | +2,03 kg |
| MAPE | 10,32% | 9,39% | +0,93 p.p. |
| Viés | +18,54 kg | +7,07 kg | +11,47 kg |
| R² | 0,496 | 0,534 | pior |

No bootstrap pareado de 10.000 iterações, a diferença de MAE teve IC95% de
-4,71 a +11,84 kg e apenas 18,8% de probabilidade de A5 ser melhor. A piora de
viés teve IC95% de +2,57 a +19,38 kg. Nos dois animais abaixo de 350 kg, o MAE
continuou muito alto, em 133,84 kg; nos outros 18, foi 27,25 kg.

**Decisão:** não promover nem repetir esta configuração em outras seeds. A
segunda vista aumentou custo e superestimação sem evidência de ganho. O suporte
multivista fica preservado para bases maiores ou protocolos calibrados.

Reprodução:

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

### A3-Gate-CowDB-001 — altura física da nuvem PLY

Antes de treinar uma fusão RGB + geometria, foi executado um gate sem consultar
os 20 animais de teste. Os PLY laterais organizados têm 512 × 424 pontos físicos
alinhados à profundidade. O fundo foi estimado somente com os 109 animais de
treino; a máscara usa diferença mínima de 150 mm, maior componente conectado e
altura entre os quantis 5% e 95% do eixo vertical.

A extração passou nos 109 animais de treino e nos 25 de validação, com 50.257 a
97.154 pontos válidos por amostra. O teste foi excluído da própria extração.

| Diagnóstico | Treino | Validação |
|---|---:|---:|
| correlação altura × peso | +0,437 | +0,402 |
| regressão só com altura — MAE | 40,03 kg | 46,89 kg |
| regressão só com altura — RMSE | 51,46 kg | 62,67 kg |

Na mesma validação, o checkpoint B2 obteve MAE de 37,63 kg. A correlação entre
altura e o resíduo real menos predito do B2 foi -0,016, praticamente zero. A
altura também apresentou correlação de apenas aproximadamente +0,36 com as
alturas manualmente medidas, evidenciando uma segmentação física ainda ruidosa.

O gate foi repetido nas outras orientações, sempre alinhando a geometria com as
predições RGB laterais do mesmo animal/evento:

| Vista geométrica | Correlação peso treino | Correlação peso validação | MAE linear validação | Correlação com resíduo B2 |
|---|---:|---:|---:|---:|
| lateral esquerda | +0,437 | +0,402 | 46,89 kg | -0,016 |
| lateral direita | +0,341 | +0,149 | 50,96 kg | -0,106 |
| superior | +0,231 | +0,258 | 49,44 kg | +0,210 |

Nenhuma vista apresentou simultaneamente poder preditivo estável e sinal
complementar forte. Como os PLY não publicam transformações extrínsecas entre
câmeras, suas coordenadas também não foram sobrepostas diretamente.

**Decisão do gate:** não integrar a altura ao EfficientNet, não gastar uma
execução GPU e não abrir o teste. Preservar o leitor PLY e a auditoria para uma
futura segmentação 3D mais fiel.

Reprodução:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_rgb_depth_point_cloud_rows.csv \
  --views left \
  --include-depth \
  --include-point-cloud
python -m ms_peso.prepare_manifest \
  --input data/interim/cowdb_rgb_depth_point_cloud_rows.csv \
  --output data/processed/rgb_depth_point_cloud_manifest.csv \
  --image-root data \
  --check-images \
  --seed 42
python -m ms_peso.audit_point_cloud_geometry \
  --manifest data/processed/rgb_depth_point_cloud_manifest.csv \
  --image-root data \
  --output artifacts/point_cloud_geometry_audit/report.json \
  --reference-checkpoint artifacts/efficientnet_b0_rgb/best_model.pt
```

## H1-Horqin-Side-Random-001 — lateral externa sem ImageNet

Experimento executado em 2026-08-23 para verificar se a base Horqin é suficiente
para treinar uma rede RGB do zero, sem reutilizar pesos ImageNet.

- fonte: Horqin versão 3, licença CC BY 4.0;
- entrada: uma imagem lateral por animal;
- dados válidos: 71 animais após excluir a lateral ausente do animal 20;
- divisão fixa, seed 42: 51 treino, 10 validação e 10 teste, sem vazamento por
  animal;
- manifesto derivado 224 × 224: SHA-256
  `6e532851cf32a416a059b1a1b52aed82af6935bb1c7673d7d6f7076412c554f9`;
- arquitetura: EfficientNet-B0;
- `pretrained: false` e `initialization: random`;
- nenhuma carga de checkpoint ou peso ImageNet;
- 100 épocas máximas, parada antecipada com paciência 15;
- melhor validação na época 18, com MAE de 44,18 kg;
- encerramento na época 33.

O conjunto de teste foi consultado somente depois da seleção da melhor época.
O resultado ficou abaixo do baseline da média:

| Métrica | EfficientNet aleatória | Média do treino |
|---|---:|---:|
| MAE | 84,54 kg | 71,40 kg |
| RMSE | 99,79 kg | 91,11 kg |
| MAPE | 18,85% | 15,62% |
| Viés | +27,50 kg | +5,16 kg |
| R² | -0,204 | -0,003 |
| Dentro de ±20 kg | 20% | 30% |

Os erros absolutos extremos ocorreram tanto em animais leves quanto pesados:
158,43 kg no animal 40, de 342 kg; 145,82 kg no animal 37, de 612 kg; e
139,72 kg no animal 17, de 371 kg. As previsões se concentraram entre
aproximadamente 461 e 549 kg, sem aprender adequadamente os extremos.

**Decisão:** experimento reprovado e `commercial_use_allowed: false`. O
checkpoint não é promovido. Cinquenta e um animais de treino não sustentaram o
ajuste de uma EfficientNet-B0 aleatória. Não selecionar uma nova arquitetura
com base nesses dez animais de teste; o próximo avanço deve vir de mais dados
laterais com pesagem real, pré-treinamento com proveniência comercial adequada
ou uma nova partição previamente congelada.

Artefatos locais ignorados pelo Git:

- `artifacts/horqin_side_random_v001/best_model.pt`, SHA-256
  `3662342a0d4947c168e367c09356f874a3341dab3eed8292d64b152dc45cdcaf`;
- `metrics.json`, SHA-256
  `ba08000f29aabb0ca46fed4e8da1929f566f5614f9297ca27e7dc8305620a58c`;
- `predictions_test.csv`, SHA-256
  `84b4e4b59384697c030847e5835fbceeb9b07d7a811d528818a7f44804b349b2`.

Reprodução:

```bash
python -m ms_peso.prepare_rgb_cache \
  --input data/processed/horqin_side_research_split.csv \
  --output data/processed/horqin_side_224_split.csv \
  --image-root data \
  --output-dir data/interim/horqin_side_224 \
  --image-size 224
python -m ms_peso.train --config configs/horqin_side_random.yaml
```

## H2-SSL-CC-BY-001 — pré-treinamento bovino sem rótulos

Gate executado em 2026-08-23 para testar uma inicialização visual permitida sem
pesos ImageNet e sem confiar nos rótulos de peso do dataset multivista.

O manifesto autossupervisionado contém somente caminhos e proveniência, sem a
coluna `weight_kg`. Foram usadas 409 imagens CC BY 4.0 únicas:

- 358 imagens do multivista após remover duas duplicatas exatas;
- 51 laterais Horqin pertencentes ao split de treino;
- zero imagens Horqin de validação ou teste;
- manifesto SHA-256
  `6ac18d7bd2e9c4c5d1b9362cb8d8534c1cf4a1d7c2346b4fb03c857de281d137`.

Uma EfficientNet-B0 aleatória foi treinada por 20 épocas com duas variações da
mesma imagem e perda NT-Xent. A perda contrastiva caiu de 3,6115 para 1,0839. O
encoder final, com `labels_used: false`, tem SHA-256
`7955fb1cc8c3489f20861f6eed8ffaae0704a068dab1085d9b0ec6625b0a74be`.

Em seguida, o encoder foi ajustado com os 51 animais Horqin de treino. O gate
consultou apenas os 10 animais de validação; o teste não foi aberto. A melhor
época foi 21 e a parada antecipada ocorreu na época 36:

| Métrica de validação | H2 SSL | H1 aleatório | Média do treino |
|---|---:|---:|---:|
| MAE | 53,81 kg | 44,18 kg | 80,99 kg |
| RMSE | 62,07 kg | 57,06 kg | 89,49 kg |
| MAPE | 11,85% | 9,94% | 17,06% |
| R² | 0,518 | 0,593 | -0,001 |

**Decisão:** o pré-treinamento aprendeu invariâncias visuais e superou a média,
mas não melhorou a inicialização aleatória na mesma validação. Gate reprovado;
não executar no teste, não promover o checkpoint e não ajustar hiperparâmetros
repetidamente nesses dez animais. Mais diversidade visual licenciada e,
principalmente, mais animais com pesagem são necessários.

Artefatos locais ignorados pelo Git:

- encoder SSL: `artifacts/ssl_cc_by_efficientnet_b0_v001/encoder_final.pt`;
- modelo do gate: `artifacts/horqin_side_ssl_validation_v001/`
  `best_validation_model.pt`, SHA-256
  `4e34fd020910eb42979f5cd0520107d89253d22be8b81af79f390c735b38d15a`;
- relatório de validação, SHA-256
  `9eac1d8926b1f91913e66c4fcde365d9e8055c1269589f336357bd891ca847e3`.

Reprodução:

```bash
python -m ms_peso.prepare_ssl_manifest \
  --input data/interim/multiview_all_224_manifest.csv \
  --input data/processed/horqin_side_224_split.csv \
  --image-root data \
  --output data/processed/ssl_cc_by_train_only.csv
python -m ms_peso.pretrain_contrastive \
  --config configs/ssl_cc_by_efficientnet_b0.yaml
python -m ms_peso.finetune_ssl_validation \
  --config configs/horqin_side_ssl_validation.yaml
```

## Ablações previstas

| ID | Mudança em relação a B1 |
|---|---|
| A1 | recorte pelo bounding box |
| A2 | fundo removido por máscara |
| A3 | escala física via PLY — gate reprovado antes do treinamento |
| A4 | metadados: raça, sexo e idade |
| A5 | fusão lateral + superior — avaliada, não promovida |
| A6 | RGB + profundidade |
| A7 | amostragem moderada por faixa de peso |

## Treinamento inicial

- pesos ImageNet;
- saída escalar, com alvo normalizado pela média/desvio do treino;
- Huber Loss;
- AdamW;
- early stopping pela MAE de validação;
- brilho/contraste e espelhamento horizontal moderados;
- sem zoom forte, pois escala corporal contém informação útil.

## Métricas mínimas

- MAE (kg);
- RMSE (kg);
- MAPE (%), quando todos os pesos forem positivos;
- R²;
- viés médio (`predito - real`);
- percentual dentro de ±5%, ±10% e ±20 kg;
- resultados por faixa de peso e pelos metadados disponíveis.

Incluir dispersão real × predito, histograma de resíduos e análise dos maiores
erros. Reportar média e intervalo de confiança por bootstrap agrupado por animal
quando o tamanho da amostra permitir.

## Validações obrigatórias

1. Teste interno com animais não vistos.
2. Teste temporal ou por lote.
3. Teste externo por fazenda antes do uso prático.
4. Comparação com baseline da média.
5. Inspeção visual (por exemplo, mapas de ativação) para procurar atalhos.

## Critério para promover um modelo

Um modelo só substitui o anterior se melhorar o erro relevante, não piorar de
forma grave nenhum subgrupo, mantiver avaliação sem vazamento e tiver custo de
inferência compatível com o dispositivo pretendido.
