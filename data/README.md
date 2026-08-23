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

## Coleta piloto própria

Os modelos em `data/templates/` iniciam o manifesto da coleta e o registro de
autorizações. Copie-os para `data/interim/`, remova as linhas de exemplo e não
versione documentos assinados ou dados pessoais. A política completa e o
comando de auditoria estão em `docs/PILOT_COLLECTION_KIT.md`.

Após aprovação, `ms_peso.seal_collection` cria em `data/processed/` um
manifesto canônico com `image_sha256` e `image_dhash`. O arquivo selado não deve
ser editado nem sobrescrito; uma correção gera um novo snapshot.
