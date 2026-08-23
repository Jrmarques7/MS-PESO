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

## Gate antes do treinamento

Nenhuma imagem entra no candidato comercial enquanto a auditoria não estiver
aprovada. Depois da coleta, ainda será necessário:

- conferir visualmente animal, pose, corpo inteiro e oclusões;
- detectar duplicatas perceptuais, além de caminhos duplicados;
- congelar o manifesto e registrar seu SHA-256;
- separar treino, validação e teste por `animal_id`;
- reservar, quando possível, outra fazenda ou período como teste externo;
- preservar distribuição de pesos, sexos, lotes e condições de captura.

A meta inicial continua sendo 100 a 200 animais, preferencialmente com mais de
uma visita e diversidade de peso. Essa quantidade inicia a validação; não é uma
garantia prévia de desempenho comercial.
