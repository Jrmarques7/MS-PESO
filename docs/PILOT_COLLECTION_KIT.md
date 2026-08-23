# Kit de coleta piloto

## Objetivo

Produzir um conjunto próprio de imagens de Nelore ligado a pesagens confiáveis
e com direitos documentados para treinamento e exploração comercial. A unidade
é o evento de pesagem; cada linha do manifesto representa o frame selecionado
de uma vista nesse evento.

## Arquivos do kit

- `configs/pilot_collection.yaml`: regras versionadas da coleta;
- `configs/image_quality.yaml`: limites técnicos da fotografia;
- `data/templates/pilot_manifest.csv`: modelo de manifesto por imagem;
- `data/templates/authorization_registry.csv`: índice das autorizações;
- `python -m ms_peso.validate_collection`: auditor automático;
- `docs/DATA_COLLECTION_PROTOCOL.md`: protocolo zootécnico e fotográfico.

As linhas dos templates são exemplos e devem ser removidas. Documentos assinados
e informações pessoais não entram no Git. Guarde-os em repositório seguro e use
somente `document_reference` para apontar ao registro correspondente.

## Preparação antes da fazenda

1. Obter instrumento de autorização revisado juridicamente.
2. Confirmar que ele cobre treinamento de modelos e uso comercial.
3. Cadastrar fazenda, câmera, balança e operador com identificadores sem dados
   pessoais desnecessários.
4. Aferir a balança e registrar `scale_id`.
5. Preparar marcador de dimensão conhecida no plano do animal.
6. Sincronizar relógios da câmera e do registro de pesagem.
7. Copiar os templates para `data/interim/`, que é ignorado pelo Git.

## Estrutura local recomendada

```text
data/raw/pilot/
  farm_001/
    nelore_0001/
      event_20260822_001/
        left.jpg
data/interim/
  pilot_manifest.csv
  authorization_registry.csv
```

Os arquivos em `data/raw` são imutáveis. Ajustes, exclusões e seleção de frames
devem ser registrados em novos manifestos, sem sobrescrever a captura original.

## Rotina por evento

1. Ler `animal_id` persistente e confirmar `farm_id` e `lot_id`.
2. Pesar na balança aferida e registrar `weight_kg` e `weighed_at`.
3. Em até 30 minutos, capturar a lateral esquerda com corpo inteiro e marcador.
4. Manter câmera aproximadamente perpendicular e um único animal dominante.
5. Registrar `captured_at`, `camera_id`, `scale_id` e eventuais anomalias.
6. Preservar os originais e selecionar somente um frame por evento/vista.
7. Marcar `quality=accepted` apenas após inspeção humana.

Horários usam ISO 8601 com fuso, por exemplo `2026-08-22T09:02:00-03:00`.
Booleanos usam exclusivamente `true` ou `false`.

## Direitos e autorização

Cada linha aponta para `authorization_id`. Para entrar na trilha comercial:

- a autorização deve pertencer à mesma fazenda;
- `status` deve ser `approved`;
- a captura precisa estar dentro da vigência;
- `allows_model_training` e `allows_commercial_use` devem ser `true`;
- `commercial_training_allowed` também deve ser `true` no manifesto;
- revogação deve impedir novas utilizações e iniciar análise jurídica dos usos
  anteriores.

`allows_data_sharing=false` não impede treinamento interno, mas proíbe publicar
ou repassar as imagens. O modelo e o relatório jurídico devem definir se algum
artefato derivado pode ser distribuído.

## Executar a auditoria

```powershell
python -m ms_peso.validate_collection `
  --manifest data/interim/pilot_manifest.csv `
  --authorizations data/interim/authorization_registry.csv `
  --image-root data/raw/pilot `
  --policy configs/pilot_collection.yaml `
  --quality-policy configs/image_quality.yaml `
  --output artifacts/collection_audit/report.json
```

Código de saída 0 significa aprovação. Código 2 significa rejeição; o JSON lista
os problemas. `--skip-image-checks` serve somente para revisar metadados durante
o preenchimento e sempre produz aviso. Ele não aprova a coleta final.

## Selar uma coleta aprovada

Quando a auditoria estiver limpa, gere um snapshot imutável:

```powershell
python -m ms_peso.seal_collection `
  --manifest data/interim/pilot_manifest.csv `
  --authorizations data/interim/authorization_registry.csv `
  --image-root data/raw/pilot `
  --output-manifest data/processed/pilot_snapshot_v001.csv `
  --output-report artifacts/collection_snapshot/v001.json
```

A selagem repete toda a auditoria com imagens, adiciona SHA-256 e dHash a cada
linha e gera um `snapshot_id` canônico. Cópias binariamente idênticas bloqueiam
o processo. Imagens visualmente muito parecidas são mantidas, mas aparecem como
alerta para revisão humana — elas podem ser visitas válidas do mesmo animal.

O relatório também registra os hashes do manifesto de origem, autorizações,
política de coleta, política de qualidade e manifesto selado. Um caminho de
saída existente nunca é sobrescrito: alterações geram `v002`, `v003` e assim
por diante.

## Criar a divisão comercial

O split comercial exige o manifesto e o relatório produzidos pela mesma
selagem. Ele recalcula todos os hashes antes de separar os animais:

```powershell
python -m ms_peso.prepare_commercial_manifest `
  --input data/processed/pilot_snapshot_v001.csv `
  --snapshot-report artifacts/collection_snapshot/v001.json `
  --image-root data/raw/pilot `
  --output data/processed/pilot_commercial_split_v001.csv `
  --output-report artifacts/commercial_split/v001.json
```

A proporção inicial é 60% treino, 15% validação, 10% calibração e 15% teste.
Todas as visitas e imagens de um `animal_id` permanecem juntas. O relatório
guarda seed, proporções, animais, imagens e distribuição de peso por partição.

- `train`: ajusta os parâmetros do modelo;
- `val`: escolhe época, arquitetura e hiperparâmetros;
- `calibration`: calibra intervalo de incerteza e regra estatística de rejeição;
- `test`: mede uma única vez o pacote final, depois de todas as escolhas.

Calibração e teste não podem fornecer exemplos, estatísticas de alvo, seleção
de época ou decisões de hiperparâmetros. Com poucos animais, a calibração será
estatisticamente grosseira; a validade do intervalo depende do tamanho e da
representatividade efetivamente coletados.

## Ajustar o candidato sem consultar os conjuntos reservados

Depois de gerar os arquivos `v001` acima, ajuste apenas treino e validação:

```powershell
python -m ms_peso.train `
  --config configs/efficientnet_b0_commercial_fit.yaml
```

Antes de inicializar o modelo, o comando confere o relatório aprovado, o hash
do manifesto, as quatro partições, as contagens e a ausência de vazamento por
animal. A arquitetura começa aleatoriamente: pesos ImageNet, B2, retomadas e
outros checkpoints iniciais são bloqueados. Durante o ajuste, somente os
arquivos de `train` e `val` têm integridade e imagem abertas.

A saída `artifacts/commercial_fit_v001/` é imutável. Ela registra apenas o
checkpoint da melhor validação, o histórico de ajuste e o manifesto resolvido
de treino/validação. Não há avaliação ou arquivo de previsões de `calibration`
ou `test`; ambos continuam lacrados para as próximas etapas.

## Calibrar a incerteza sem consultar o teste

Depois que o ajuste terminar, calcule o hash que identifica seu checkpoint:

```powershell
(Get-FileHash `
  artifacts/commercial_fit_v001/best_model.pt `
  -Algorithm SHA256).Hash.ToLower()
```

Copie o resultado para `model.checkpoint_sha256` em
`configs/efficientnet_b0_commercial_calibration.yaml` e execute:

```powershell
python -m ms_peso.calibrate `
  --config configs/efficientnet_b0_commercial_calibration.yaml
```

O comando autentica o checkpoint, o manifesto, o snapshot e o relatório do
split antes de abrir imagens. Somente `calibration` é carregado. O método usa
split conformal com erro absoluto e agrupa por `animal_id`: quando existem
várias imagens ou visitas, o pior erro do animal é seu único escore. Isso evita
uma falsa multiplicação da amostra e produz um intervalo mais conservador.

A cobertura padrão é 90%, que exige no mínimo nove animais independentes de
calibração para que o quantil finito exista. Se a coleta não sustentar a
cobertura configurada, a execução falha sem criar artefatos. A saída imutável
contém `calibration.json`, `predictions_calibration.csv` e
`resolved_calibration_manifest.csv`. Nenhuma imagem ou métrica de `test` é
produzida, e o modelo permanece `not_promoted`.

## Gate antes do treinamento

Nenhuma imagem entra no candidato comercial enquanto a auditoria não estiver
aprovada. Depois da coleta, ainda será necessário:

- conferir visualmente animal, pose, corpo inteiro e oclusões;
- revisar os pares perceptualmente semelhantes apontados pela selagem;
- congelar o manifesto e registrar seu SHA-256;
- separar treino, validação e teste por `animal_id`;
- reservar calibração independente sem compartilhamento de `animal_id`;
- reservar, quando possível, outra fazenda ou período como teste externo;
- preservar distribuição de pesos, sexos, lotes e condições de captura.

A meta inicial continua sendo 100 a 200 animais, preferencialmente com mais de
uma visita e diversidade de peso. Essa quantidade inicia a validação; não é uma
garantia prévia de desempenho comercial.
