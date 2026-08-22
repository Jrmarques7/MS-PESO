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

## Ablações previstas

| ID | Mudança em relação a B1 |
|---|---|
| A1 | recorte pelo bounding box |
| A2 | fundo removido por máscara |
| A3 | marcador de escala/câmera calibrada |
| A4 | metadados: raça, sexo e idade |
| A5 | fusão lateral + traseira/superior |
| A6 | RGB + profundidade |

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
