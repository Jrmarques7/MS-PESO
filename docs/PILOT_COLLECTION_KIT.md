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
- `data/templates/pasture_video_manifest.csv`: inventário bruto de vídeos no pasto;
- `data/templates/authorization_registry.csv`: índice das autorizações;
- `python -m ms_peso.init_collection_batch`: cria um lote vazio sem sobrescrever;
- `python -m ms_peso.select_collection_frames`: escolhe um quadro técnico por
  evento/vista sem executar o modelo de peso;
- `python -m ms_peso.validate_collection`: auditor automático;
- `docs/DATA_COLLECTION_PROTOCOL.md`: protocolo zootécnico e fotográfico.
- `docs/PASTURE_CAPTURE_PLAN.md`: execução progressiva no cenário real de pasto.

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
7. Criar o lote vazio com o comando abaixo; `data/raw` e `data/interim` são
   ignorados pelo Git.

```powershell
python -m ms_peso.init_collection_batch `
  --batch-id batch_20260823_001 `
  --farm-id farm_001
```

O comando gera manifestos contendo apenas os cabeçalhos e recusa reutilizar um
`batch_id`. As linhas de exemplo dos templates não são copiadas.

## Estrutura local recomendada

```text
data/raw/pasture/
  farm_001/
    nelore_0001/
      event_20260822_001/
        identity.jpg
        lateral_left_01.mp4
data/interim/pasture/
  batch_20260823_001/
    batch_metadata.json
    pasture_video_manifest.csv
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

## Selecionar o melhor quadro dos vídeos

Primeiro, uma pessoa confere cada vídeo bruto e só marca `quality=accepted`,
`primary_full_body=true` e `primary_lateral=true` quando identidade, corpo e
pose estiverem corretos. Em seguida:

```powershell
python -m ms_peso.select_collection_frames `
  --manifest data/interim/pasture/batch_20260823_001/pasture_video_manifest.csv `
  --video-root data/raw/pasture `
  --image-root data `
  --output-directory data/interim/pasture/batch_20260823_001/selection_v001
```

O processo avalia resolução, exposição e nitidez, compara todas as tomadas do
mesmo evento/vista e materializa somente o melhor quadro. Ele copia
`weight_kg` do manifesto de balança sem fazer inferência e nunca altera o vídeo
original. A saída contém `pilot_manifest.csv`, `selected_frames/` e
`selection_report.json`.

Todo quadro gerado recebe novamente `quality=review`, pois a seleção automática
não confirma bovino, identidade, lateralidade, corpo inteiro ou oclusão. Após
inspeção visual, copie o manifesto para `pilot_manifest_reviewed_v001.csv` e
promova somente os quadros corretos para `accepted`. Uma nova execução usa
`selection_v002`; saídas existentes não são sobrescritas.

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
  --manifest data/interim/pasture/batch_20260823_001/selection_v001/pilot_manifest_reviewed_v001.csv `
  --authorizations data/interim/pasture/batch_20260823_001/authorization_registry.csv `
  --image-root data `
  --policy configs/pilot_collection.yaml `
  --quality-policy configs/image_quality.yaml `
  --output artifacts/collection_audit/batch_20260823_001_v001.json
```

Código de saída 0 significa aprovação. Código 2 significa rejeição; o JSON lista
os problemas. `--skip-image-checks` serve somente para revisar metadados durante
o preenchimento e sempre produz aviso. Ele não aprova a coleta final.

## Selar uma coleta aprovada

Quando a auditoria estiver limpa, gere um snapshot imutável:

```powershell
python -m ms_peso.seal_collection `
  --manifest data/interim/pasture/batch_20260823_001/selection_v001/pilot_manifest_reviewed_v001.csv `
  --authorizations data/interim/pasture/batch_20260823_001/authorization_registry.csv `
  --image-root data `
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
  --image-root data `
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

## Abrir o teste uma única vez

Calcule também o hash do relatório de calibração:

```powershell
(Get-FileHash `
  artifacts/commercial_calibration_v001/calibration.json `
  -Algorithm SHA256).Hash.ToLower()
```

Preencha os dois hashes e todos os limites marcados como
`REPLACE_BEFORE_OPENING_TEST` em
`configs/efficientnet_b0_commercial_evaluation.yaml`. Esses limites precisam
vir do uso operacional e ser aprovados antes de qualquer resultado de teste.
Em seguida, execute uma única vez:

```powershell
python -m ms_peso.evaluate_commercial `
  --config configs/efficientnet_b0_commercial_evaluation.yaml
```

Antes de tocar nas imagens, o processo recusa calibração com intervalo largo
demais, quantidade insuficiente de animais, hashes divergentes ou critérios
incompletos. Quando a abertura é permitida, cria de forma atômica o recibo
`artifacts/commercial_test_access/v001.json`. A existência desse recibo bloqueia
outra tentativa, inclusive com diretório de resultados diferente.

O teste produz métricas pontuais por imagem, métricas balanceadas por animal
com bootstrap de 95%, cobertura conformal com intervalo de Wilson,
`predictions_test.csv` e o relatório final. Os limites de confiança, e não só
os valores pontuais, determinam o gate técnico. Uma falha consome o teste e não
autoriza ajustar limites usando seus resultados; um novo candidato exige teste
independente.

Mesmo uma passagem completa gera apenas recomendação para revisão. Direitos,
domínio externo, segurança operacional e aprovação humana permanecem
obrigatórios, e `commercial_use_allowed` continua `false`.

## Montar o pacote de inferência interna

Somente quando o relatório final indicar `technical_review_recommended`, copie
os hashes reais do checkpoint, da calibração e da avaliação para
`models/commercial_candidate.yaml`. O descritor também fixa a política de
qualidade e o model card; qualquer alteração posterior invalida o pacote.

Para executar uma foto lateral em ambiente interno:

```powershell
python -m ms_peso.predict_commercial `
  --image caminho/para/foto_lateral.jpg `
  --package models/commercial_candidate.yaml `
  --device auto
```

O gate de resolução, proporção, exposição e nitidez roda antes da carga do
checkpoint. Uma rejeição retorna peso e intervalo nulos. Uma captura aceita
retorna estimativa, raio conformal, limites inferior/superior e cobertura-alvo.
O limite inferior é truncado em zero sem excluir pesos fisicamente possíveis.

Esse pacote é exclusivamente interno e continua bloqueado para exploração
comercial. Ele não confirma bovino, raça, pose lateral, corpo inteiro, oclusão
ou domínio validado. O adaptador HTTP do ponto 7 vive separado desse núcleo no
pacote `ms_peso.service`, não conhece os conceitos de fazenda ou lote e recusa o
candidato por padrão. A aplicação consumidora será integrada somente depois.

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
