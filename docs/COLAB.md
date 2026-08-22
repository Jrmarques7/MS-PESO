# Execução no Google Colab

## Opção A — repositório Git remoto

Depois de publicar o projeto em um repositório acessível:

```python
!git clone URL_DO_REPOSITORIO ms-peso
%cd ms-peso
!pip install -q -e .
```

## Opção B — upload compactado

Compacte o projeto sem `data/raw` e faça upload para o Drive. Em seguida,
descompacte, entre no diretório e instale com `pip install -e .`.

## Dados persistentes no Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Mantenha `data/raw/cowdb` e `data/processed/manifest.csv` no Drive. Os caminhos
do manifesto são relativos a `data`, portanto a configuração existente funciona
quando essa estrutura é preservada. A cópia completa do CowDB ocupa cerca de
5,3 GB.

## Verificações

```python
!python -m ms_peso.prepare_manifest \
  --input data/interim/cowdb_manifest.csv \
  --output data/processed/manifest.csv \
  --image-root data \
  --check-images

!pytest -q
```

No menu do Colab, selecione um ambiente com GPU e confirme:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Treinamento

```python
!python -m ms_peso.train --config configs/baseline_rgb.yaml
```

Antes do B1, reproduza o piso B0 no mesmo manifesto:

```python
!python -m ms_peso.evaluate_mean_baseline \
  --manifest data/processed/manifest.csv \
  --image-root data \
  --view left \
  --check-images
```

Copie `artifacts/` para o Drive ao final da sessão. O modelo só deve ser
comparado com outro se ambos usarem o mesmo manifesto e a mesma partição por
animal.
