# Princípios de engenharia

## Regra obrigatória: Single Responsibility Principle

O **SRP é uma das regras principais e não negociáveis do MS-PESO**.

Cada módulo, classe e função deve ter uma responsabilidade coesa e apenas um
motivo principal para mudar. “Responsabilidade” significa o ator, requisito ou
tipo de decisão que provoca uma alteração — não simplesmente a quantidade de
linhas do componente.

## Fronteiras esperadas

| Componente | Responsabilidade |
|---|---|
| `manifest.py` | contrato, validação e partição do manifesto |
| `dataset.py` | carregar uma amostra e aplicar transformações visuais |
| `model.py` | construir arquiteturas de regressão |
| `metrics.py` | calcular métricas sem conhecer treino ou armazenamento |
| `training.py` | executar épocas e early stopping sem persistir artefatos |
| `evaluation.py` | executar e representar a avaliação do modelo |
| `artifacts.py` | persistir checkpoints, relatórios e predições |
| `train.py` | somente orquestrar o caso de uso de treinamento |
| `model_package.py` | validar identidade, integridade e domínio do artefato |
| `image_quality.py` | medir a captura e aplicar uma política de qualidade |
| `inference.py` | carregar o modelo congelado e estimar uma imagem |
| `predict.py` | somente orquestrar a inferência pela linha de comando |
| `importers/` | converter uma fonte externa para o contrato MS-PESO |
| futuros módulos de inferência | carregar o artefato e produzir estimativas |
| futuros módulos de coleta | adquirir imagens e metadados de campo |

Um módulo de orquestração pode chamar as responsabilidades acima, mas não deve
reimplementar suas regras.

## Exemplos

Correto:

- a validação do CSV fica no módulo do manifesto;
- uma métrica recebe valores reais e previstos e retorna resultados;
- o modelo define somente a rede e sua cabeça de regressão;
- a interface chama um serviço de inferência já testável isoladamente.

Incorreto:

- uma classe que baixa dados, treina a rede e gera gráficos;
- uma função de dataset que também decide os splits;
- a API contendo pré-processamento, arquitetura e persistência do modelo;
- um notebook se tornar a única implementação do pipeline.

## Critérios para revisão

Antes de aceitar uma mudança, responder:

1. Qual é a responsabilidade do componente alterado?
2. Existe mais de um motivo independente para ele mudar?
3. A lógica pode ser testada sem inicializar responsabilidades não relacionadas?
4. O nome do componente descreve claramente o que ele faz?
5. A composição/orquestração está separada das regras específicas?

Se houver dois motivos independentes para mudança, o componente deve ser
separado. Exceções precisam ser justificadas no registro de decisões, com plano
explícito para remoção quando forem temporárias.

## Aplicação em notebooks

Notebooks são permitidos para exploração, visualização e demonstração. Toda
lógica reutilizável deve ficar em `src/ms_peso/`; o notebook apenas importa e
orquestra essas funções. Isso mantém o experimento reproduzível e evita uma
segunda implementação do pipeline.
