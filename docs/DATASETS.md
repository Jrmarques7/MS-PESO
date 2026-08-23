# Datasets

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
As nuvens de pontos ficam reservadas para experimentos posteriores.

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

Depois de dividir esse manifesto, recortes ou máscaras retangulares podem ser
gerados com `python -m ms_peso.prepare_depth_crops`. O fundo é calculado apenas
com linhas `train`; imagens brutas não são modificadas. A ferramenta exige uma
caixa mínima de 50% da cena e registra parâmetros e coordenadas no manifesto
derivado. Esses derivados são experimentais e não substituem o manifesto RGB.

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
