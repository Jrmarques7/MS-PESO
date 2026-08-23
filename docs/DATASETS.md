# Datasets

## Bases Mendeley auditadas em 2026-08-23

Duas bases com licença `CC BY 4.0` foram baixadas diretamente da fonte,
verificadas por SHA-256 e inspecionadas imagem a imagem. A licença permite uso
comercial com atribuição, mas isso não torna automaticamente um modelo apto ao
uso comercial: representatividade, qualidade dos rótulos e validação no domínio
FarmUp continuam obrigatórias.

Os ZIPs ficam em `data/raw/mendeley/archives/` e não são versionados. A extração
é reproduzível, confirma tamanho e hash e rejeita caminhos inseguros:

```bash
python -m ms_peso.extract_mendeley_cattle --dataset multiview
python -m ms_peso.extract_mendeley_cattle --dataset horqin
```

### Horqin lateral e traseiro

Fonte oficial: <https://data.mendeley.com/datasets/h2s22wr5py/3>

É a fonte externa mais próxima da captura lateral pretendida no pasto. A versão
3 contém medidas e pesos de 72 bovinos Horqin entre 341 e 644 kg. A auditoria do
arquivo encontrou 71 imagens laterais e 72 traseiras, todas decodificáveis.

Três problemas foram registrados sem alterar os dados brutos:

- a lateral do animal 20 está ausente;
- a traseira 50 tem somente 750 × 1000 pixels, contra 3024 × 4032 nas demais;
- as traseiras 57 e 67 são exatamente o mesmo arquivo.

Para o primeiro experimento lateral, o importador exclui explicitamente apenas
o animal 20 e produz 71 linhas. Ele se recusa a continuar se a opção de exclusão
não for fornecida:

```bash
python -m ms_peso.import_horqin \
  --dataset-root data/raw/mendeley/horqin_v3 \
  --image-root data \
  --output data/interim/horqin_side_manifest.csv \
  --views side \
  --exclude-known-anomalies
```

Ao importar lateral e traseira juntas, os animais 20, 50, 57 e 67 são excluídos
por completo, deixando 68 animais e 136 imagens pareadas. A divisão deve ser
feita somente depois, sempre por `animal_id`, para impedir vazamento entre
treino, validação e teste.

O manifesto lateral local foi validado com Pillow e dividido com seed 42 em 51
animais de treino, 10 de validação e 10 de teste. Não há animal compartilhado
entre as partições. O SHA-256 do arquivo reproduzível
`data/processed/horqin_side_research_split.csv` é
`c6be975eb74188d1a974791c557cede0f971a576d0da87a0ed30a9b0aba0a993`.

Para evitar reabrir os PNGs de 12 MP em toda época, o redimensionamento
determinístico para 224 × 224 foi materializado em `data/interim/`, mantendo a
imagem original registrada em `source_image_path`. O manifesto derivado conserva
os mesmos animais e splits e tem SHA-256
`6e532851cf32a416a059b1a1b52aed82af6935bb1c7673d7d6f7076412c554f9`.

O primeiro treinamento sem ImageNet foi executado e reprovado: a
EfficientNet-B0 inicializada aleatoriamente obteve MAE de 84,54 kg no teste,
pior que os 71,40 kg do baseline da média. O resultado completo está em
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md) e não autoriza promoção comercial.

Também foi testado pré-treinamento contrastivo sem rótulos com 358 imagens
multivista únicas e somente as 51 laterais Horqin de treino. O ajuste posterior
obteve MAE de validação de 53,81 kg, pior que os 44,18 kg da inicialização
aleatória na mesma validação. O teste não foi reaberto. Essa linha também foi
encerrada sem promoção.

Horqin é útil para pré-treinamento, ensaios de arquitetura e validação do
pipeline. Não substitui a coleta própria: raça, fazenda, câmera, vegetação,
distância, postura e distribuição de peso diferem do cenário brasileiro.

### Multivista com cinco ângulos numéricos

Fonte oficial: <https://data.mendeley.com/datasets/vf7pxfs7dx/1>

A versão 1 contém 72 animais, 360 JPEGs e cinco imagens por animal. Todas abrem
corretamente, porém a auditoria encontrou pesos declarados entre 40 e 1.300 kg,
mediana de 145 kg, sete rótulos abaixo de 80 kg e cinco acima de 900 kg. Também
há dois pares de imagens duplicadas do animal 57. A fonte não documenta qual
número corresponde semanticamente a lateral esquerda, direita, frente ou
traseira, e a amostra visual inclui baias e ambientes de mercado, não apenas
pasto livre.

Por isso, esta base permanece em quarentena. O importador exige confirmação
explícita, omite nomes de coletores e GPS exato e grava
`training_eligible=false`, `quality=review_required` e
`label_status=unverified_source_label`:

```bash
python -m ms_peso.import_mendeley_multiview \
  --dataset-root data/raw/mendeley/multiview_v1 \
  --image-root data \
  --output data/interim/multiview_review_manifest.csv \
  --angles angle_1 \
  --acknowledge-unverified-source-labels
```

Esse manifesto serve para auditoria e engenharia multivista. Ele não deve ser
promovido a conjunto comercial aprovado sem validação independente dos pesos e
documentação dos ângulos.

## Estratégia brasileira

As fontes brasileiras e específicas de Nelore são prioritárias para o domínio
final. O inventário, disponibilidade, ressalvas e ordem operacional estão em
[Fontes brasileiras de dados bovinos](BRAZILIAN_DATA_SOURCES.md) e no registro
estruturado [`data/source_registry.yaml`](../data/source_registry.yaml).

CowDB permanece como base pública de engenharia para validar o pipeline, mas
não representa a raça nem as condições brasileiras pretendidas.

O estudo Embrapa/UFGD de 2026 é a fonte UAV brasileira com peso mais concreta
encontrada até agora: usa aproximadamente 190 imagens para reconstrução 3D no
mesmo dia da pesagem de um lote com 70 Nelore e relata RMSE de 8,35 kg. Os
arquivos não foram publicados, o total final de animais avaliados não está
claro e apenas sete animais calibraram a regressão; por isso, ele entra como
fonte P0 para solicitação, não como resultado diretamente comparável ao
baseline RGB atual.

## NelloreBeefCattleDataset longitudinal

Fonte oficial:
<https://github.com/EvertonTetila/NelloreBeefCattleDataset>

O ZIP público foi auditado em 2026-08-22 por leitura do diretório central e de
arquivos internos pequenos via intervalos HTTP, sem baixar os 9,68 GB. Ele
contém 904 imagens UAV brutas em doze datas, 232 imagens pareadas com rótulos
YOLO/LabelMe de `cattle-back` e `cattle-head`, e 23 pares LabelMe para o cocho.

Não há peso, `animal_id` persistente, tabela de pesagem nem licença explícita no
arquivo. Por isso, essa versão é auxiliar e nunca deve ser passada ao importador
do manifesto de regressão.

A busca pelo conjunto de aproximadamente 10 mil amostras e 110 animais relatado
no workshop de 2025 chegou ao mesmo repositório público já auditado. A promessa
de disponibilização não é reproduzível com os arquivos atuais e precisa ser
esclarecida com os autores.

Após extraí-la em um volume externo com espaço suficiente, valide sua cópia:

```bash
python -m ms_peso.inspect_nellore_uav \
  --dataset-root /caminho/NelloreBeefCattleDataset \
  --output artifacts/nellore_uav_inventory.json
```

O relatório verifica sessões, pares imagem/JSON/YOLO, classes e arquivos que
possam ser candidatos a metadados. A presença futura de um CSV com peso ainda
não basta: será necessário auditar o vínculo individual imagem-pesagem.

## CowDB

Fonte oficial: <https://github.com/ruchaya/CowDB>

Auditoria realizada em 2026-08-22 sobre a árvore `master`:

- 154 bovinos Hereford identificados por diretórios `1` a `154`;
- uma captura por animal;
- vistas `left`, `right` e `top`;
- para cada vista: uma imagem RGB PNG e uma imagem de profundidade PNG;
- uma nuvem de pontos PLY por vista;
- `Manual_measurements.xlsx` com peso vivo e nove medidas corporais;
- peso observado na planilha entre 243 e 605 kg;
- 1.388 arquivos rastreados na revisão auditada.

A cópia local usada no primeiro manifesto está fixada no commit
`270f2908de9a6931789400fbea122fe5e8df35b6`. Ela ocupa aproximadamente 5,3 GB.

O importador inicial usa somente `raw/<view>/rgb-*.png`. Com
`--include-depth`, ele também valida o par `depth-*.png` com o mesmo timestamp
e grava `depth_image_path`; a importação falha se qualquer par estiver ausente.
Com `--include-point-cloud`, ele valida e grava também o PLY organizado da
mesma vista e timestamp em `point_cloud_path`.

### Licença

O README afirma que a base é aberta e acessível à comunidade científica, mas a
revisão auditada não contém um arquivo `LICENSE` nem termos explícitos de
redistribuição. Portanto:

- o MS-PESO não redistribui imagens nem a planilha;
- cada usuário obtém sua própria cópia da fonte oficial;
- publicação de cópias, derivados ou modelos deve aguardar confirmação dos
  termos com os autores ou outra base jurídica adequada;
- registrar o commit/revisão usada em cada experimento.

### Conversão

Obtenha a base diretamente da fonte oficial:

```bash
git clone --depth 1 https://github.com/ruchaya/CowDB.git data/raw/cowdb
```

Converta a cópia local:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_manifest.csv \
  --views left
```

Para preparar o manifesto multimodal:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_rgb_depth_manifest.csv \
  --views left \
  --include-depth
```

Para incluir a nuvem física já calculada com os parâmetros internos da câmera:

```bash
python -m ms_peso.import_cowdb \
  --dataset-root data/raw/cowdb \
  --image-root data \
  --output data/interim/cowdb_rgb_depth_point_cloud_rows.csv \
  --views left \
  --include-depth \
  --include-point-cloud
```

Depois de dividir esse manifesto, recortes ou máscaras retangulares podem ser
gerados com `python -m ms_peso.prepare_depth_crops`. O fundo é calculado apenas
com linhas `train`; imagens brutas não são modificadas. A ferramenta exige uma
caixa mínima de 50% da cena e registra parâmetros e coordenadas no manifesto
derivado. Esses derivados são experimentais e não substituem o manifesto RGB.

Para a entrada multivista, importe `--views left top` e execute
`python -m ms_peso.prepare_multi_view_manifest`. O preparador exige exatamente
uma imagem de cada vista por animal/evento, confirma peso e partição iguais e
grava o caminho superior em `secondary_image_path`. A divisão continua sendo
feita depois do pareamento, por animal, com `prepare_manifest`.

O importador lê a grafia original `live weithg`, preserva as medidas corporais
como metadados e produz identificadores isolados (`cowdb_001`, etc.). Ele falha
se houver peso sem imagem, imagem sem peso, vista desconhecida ou arquivo fora
de `image_root`.

### Manifesto validado

O manifesto RGB lateral esquerdo foi gerado e todas as 154 imagens foram
verificadas com Pillow. A divisão reproduzível com seed 42 ficou assim:

| Split | Animais | Imagens | Peso mínimo | Peso médio | Peso máximo |
|---|---:|---:|---:|---:|---:|
| treino | 109 | 109 | 275 kg | 452,2 kg | 565 kg |
| validação | 25 | 25 | 312 kg | 454,6 kg | 605 kg |
| teste | 20 | 20 | 243 kg | 450,8 kg | 573 kg |

Não há animal compartilhado entre partições.

O manifesto lateral + superior também contém 154 eventos e conserva exatamente
os mesmos 109/25/20 animais em cada partição. Seu SHA-256 é
`cac0b63ba20aa3853f5c43c679b7bca92761780118905ba5386c8d27392a65f1`.

A nuvem PLY possui 217.088 vértices binários little-endian, correspondentes aos
512 × 424 pixels da profundidade. As coordenadas `x`, `y` e `z` já estão em
unidades físicas; o bloco de câmera do arquivo contém valores padrão do PCL e
não deve ser tratado como uma calibração publicada separadamente.
