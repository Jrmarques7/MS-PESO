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
