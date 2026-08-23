# Dados

Imagens brutas e artefatos processados não são versionados neste repositório.
O código consome um manifesto CSV, uma linha por imagem selecionada.

## Esquema mínimo

```csv
image_path,animal_id,event_id,weight_kg,view,breed,sex,farm_id,quality
raw/cowdb/001/left.jpg,cow_001,weigh_001,412.5,left,hereford,female,farm_public,accepted
```

Os quatro primeiros campos são obrigatórios. Para o baseline, mantenha uma
imagem por evento e vista. No arquivo `configs/baseline_rgb.yaml`, `image_root`
é `data`; portanto `raw/...` resolve para `data/raw/...`. Sem `image_root`, um
caminho relativo é resolvido a partir do diretório do manifesto.

Experimentos multimodais podem incluir `depth_image_path`. A configuração deve
declarar `depth_image_column: depth_image_path`; nesse caso, o treinamento
valida e carrega os dois arquivos. O caminho de profundidade é opcional no
contrato geral e não altera o baseline RGB.

Fontes 3D podem incluir `point_cloud_path`. Esse campo aponta para um arquivo
de nuvem de pontos, não para uma imagem; por isso ele recebe validação própria.
No CowDB, `--include-point-cloud` associa explicitamente cada PLY à vista e ao
timestamp correspondentes.

Depois de executar `ms_peso.prepare_manifest`, a coluna `split` terá um dos
valores `train`, `val` ou `test`.

## Diretórios

```text
data/raw/          cópia imutável do dataset original
data/interim/      manifesto ainda não dividido e arquivos em auditoria
data/processed/    manifesto pronto para experimento
```

Nunca edite `raw` para corrigir um exemplo. Registre a exclusão ou correção em
um novo manifesto processado.

Os datasets Mendeley auditados usam um estágio intermediário adicional. Os ZIPs
originais e suas extrações permanecem em `data/raw/mendeley/`; os importadores
gravam apenas manifestos em `data/interim/`. O Horqin exige aceite explícito das
anomalias conhecidas, e o multivista gera somente um manifesto de revisão em
quarentena. Consulte `docs/DATASETS.md` e `data/source_registry.yaml` antes de
promover qualquer fonte externa.

## Coleta piloto própria

Os modelos em `data/templates/` definem o manifesto da coleta e o registro de
autorizações. Use `ms_peso.init_collection_batch` para criar cópias vazias em
`data/interim/pasture/<batch_id>/`; ele não copia as linhas de exemplo e não
sobrescreve um lote existente. Documentos assinados e dados pessoais não são
versionados.

Para vídeos laterais, `ms_peso.select_collection_frames` escolhe somente um
quadro técnico por evento/vista e o grava junto de um manifesto rastreável. O
processo não estima peso: `weight_kg` é copiado do registro da balança. Cada
quadro selecionado retorna a `quality=review` e precisa de inspeção humana. A
política completa e os comandos estão em `docs/PILOT_COLLECTION_KIT.md`.

Após aprovação, `ms_peso.seal_collection` cria em `data/processed/` um
manifesto canônico com `image_sha256` e `image_dhash`. O arquivo selado não deve
ser editado nem sobrescrito; uma correção gera um novo snapshot.

`ms_peso.prepare_commercial_manifest` verifica novamente imagens, manifesto e
relatório de selagem antes de criar `train`, `val`, `calibration` e `test`. O
split histórico de pesquisa com três partições permanece disponível.
