# Protocolo de coleta de dados

## 1. Unidade experimental

A unidade é o **evento de pesagem**, não a fotografia. Um evento relaciona um
animal, um peso de referência e uma ou mais imagens capturadas em intervalo
curto. Vários frames do mesmo vídeo não contam como novas observações
independentes.

## 2. Preparação

- identificar o animal de forma persistente;
- aferir/zerar a balança conforme orientação do fabricante;
- fixar câmera, altura, orientação e distância quando possível;
- posicionar marcador de tamanho conhecido no plano do animal;
- registrar identificadores de fazenda, instalação, câmera e sessão;
- garantir passagem segura, sem aumentar estresse ou risco aos animais.

## 3. Captura por evento

1. Registrar `animal_id` e `event_id`.
2. Obter o peso real e horário da pesagem.
3. Capturar lateralmente o corpo inteiro, sem oclusões importantes.
4. Se previsto, capturar vista traseira ou superior na mesma sessão.
5. Registrar metadados e qualquer anomalia.
6. Não apagar exemplos difíceis; marque sua qualidade.

Recomendação inicial: selecionar um frame de boa qualidade por vista e evento.
Os vídeos originais podem ser preservados fora do repositório para estudos
posteriores.

## 4. Campos obrigatórios

| Campo | Descrição |
|---|---|
| `image_path` | Caminho relativo ou absoluto da imagem |
| `animal_id` | Identificador persistente do bovino |
| `event_id` | Identificador único da sessão/pesagem |
| `weight_kg` | Peso de referência em quilogramas |

## 5. Campos recomendados

| Campo | Exemplo |
|---|---|
| `view` | `left`, `right`, `rear`, `top` |
| `breed` | `nelore` |
| `sex` | `male`, `female` |
| `age_months` | `18` |
| `farm_id` | `farm_01` |
| `lot_id` | `lot_2026_08` |
| `captured_at` | ISO 8601 com fuso horário |
| `camera_id` | modelo/dispositivo persistente |
| `distance_m` | distância aproximada da câmera |
| `quality` | `accepted`, `blur`, `occluded`, `bad_pose` |
| `scale_marker` | `true`/`false` |
| `body_condition_score` | valor e escala adotada |
| `notes` | observações livres |

## 6. Controle de qualidade

Uma imagem aceita no baseline deve mostrar cabeça/tronco/quartos e patas sem
cortes relevantes; ter um único animal dominante; possuir foco suficiente; e
apresentar pose aproximadamente lateral. A decisão de aceitar/rejeitar deve ser
guardada, não aplicada silenciosamente.

Verificações automáticas antes do treino:

- caminhos existem e imagens podem ser abertas;
- pesos são numéricos, positivos e plausíveis para a população;
- `animal_id` e `event_id` não estão vazios;
- um evento não possui pesos conflitantes;
- imagens duplicadas são sinalizadas;
- cada animal pertence a somente um split.

## 7. Divisão dos dados

- `train`: aproximadamente 70% dos animais;
- `val`: aproximadamente 15%;
- `test`: aproximadamente 15%;
- agrupamento obrigatório por `animal_id`;
- estratificação aproximada por peso quando houver animais suficientes;
- balanceamento ou ponderação somente em `train`;
- `test` mantém a distribuição natural do cenário de uso.

Quando possível, criar ainda um teste externo contendo outra fazenda, sessão ou
período. Ele é mais informativo sobre uso real que uma divisão aleatória.

## 8. Governança

- documentar consentimento/autorização da fazenda;
- evitar informações pessoais desnecessárias;
- manter dados brutos imutáveis e com backup;
- versionar manifestos, não arquivos pesados, no Git;
- registrar correções de rótulos;
- definir licença antes de publicar imagens ou modelos.

