# Model card — B2 CowDB EfficientNet-B0

## Identificação

- `model_id`: `b2-cowdb-efficientnet-b0`;
- versão do pacote: `1`;
- estado: experimental;
- pronto para produção: não;
- arquitetura: EfficientNet-B0 com pesos iniciais ImageNet;
- entrada: uma imagem RGB lateral esquerda, redimensionada para 224 × 224;
- saída: estimativa escalar de peso vivo em quilogramas;
- melhor época da execução principal: 8.

## Finalidade permitida

Este modelo serve para validar tecnicamente carregamento, inferência, avaliação
e integração de software em cenário de pesquisa. Pode ser usado em testes
locais e demonstrações claramente identificadas como experimentais.

## Usos não permitidos

- estimar peso de Nelore como se o modelo estivesse validado nessa raça;
- apoiar venda, compra ou pagamento por peso;
- definir dosagem veterinária, dieta ou procedimento clínico;
- substituir balança aferida;
- operar silenciosamente fora do protocolo de captura;
- apresentar a saída como medição certificada ou modelo de produção.

## Dados de desenvolvimento

O modelo foi desenvolvido com o CowDB, contendo 154 bovinos Hereford de uma
fazenda na Rússia. Cada animal possui uma captura; o baseline usa somente a
vista lateral esquerda. A divisão fixa por animal contém:

| Partição | Animais |
|---|---:|
| treino | 109 |
| validação | 25 |
| teste | 20 |

Nenhum `animal_id` aparece em mais de uma partição. O CowDB valida o pipeline,
mas não representa a raça Nelore nem o ambiente brasileiro pretendido.

## Treinamento

- alvo normalizado pela média e desvio-padrão do treino;
- Huber Loss;
- AdamW;
- aumentos moderados de cor e espelhamento horizontal;
- parada antecipada pela MAE de validação;
- teste consultado somente após selecionar o checkpoint pela validação.

## Desempenho

Execução principal, seed 42, sobre 20 animais de teste nunca vistos:

| Métrica | Resultado |
|---|---:|
| MAE | 34,21 kg |
| RMSE | 51,06 kg |
| MAPE | 9,39% |
| Viés médio | +7,07 kg |
| R² | 0,534 |
| dentro de ±10 kg | 20% |
| dentro de ±20 kg | 45% |

Em três seeds, o B2 obteve MAE de 32,10 ± 2,43 kg, RMSE de 49,11 ± 4,36 kg e
MAPE de 8,84 ± 0,73%. Esses números medem variação entre treinamentos sobre a
mesma divisão pequena; não constituem validação externa.

## Limitações conhecidas

- apenas 154 animais e 20 no teste;
- somente Hereford e uma origem;
- somente captura lateral controlada;
- nenhuma validação externa por fazenda, câmera ou período;
- os dois animais de teste abaixo de 350 kg tiveram MAE aproximado de 133 kg;
- tendência a regressão para a média em faixas raras;
- gate técnico cobre resolução, exposição e nitidez, mas não confirma animal,
  pose, vista lateral, corpo inteiro ou oclusão;
- saída pontual sem intervalo individual de incerteza calibrado.

## Requisitos de captura para demonstração

- um único bovino dominante;
- corpo inteiro visível;
- vista lateral esquerda aproximadamente perpendicular;
- foco e exposição adequados;
- câmera e distância semelhantes às condições controladas;
- não interpretar a saída se a captura estiver fora desses requisitos.

## Integridade e rastreabilidade

O descritor versionado está em `models/b2_cowdb.yaml`. O checkpoint local
esperado é `artifacts/efficientnet_b0_rgb/best_model.pt`, ignorado pelo Git, com
SHA-256:

```text
43fe4b97c65ffcafd1e68d8e6dcfac6059c807019898120bf6ecbfa34541dedd
```

A inferência recusa o arquivo se o hash, a arquitetura, o tamanho de entrada ou
os metadados de normalização divergirem do pacote.

## Próxima condição para promoção

O modelo não será promovido diretamente. É necessário coletar dados de Nelore
com pesagem vinculada, criar teste externo, definir erro operacional aceitável
e demonstrar desempenho e segurança no domínio brasileiro.
