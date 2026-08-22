# Registro de decisões

## 2026-08-22 — Separar o ZIP UAV longitudinal da base de peso de 2025

**Decisão:** classificar o ZIP público atual do NelloreBeefCattleDataset como
fonte auxiliar P1 e manter a base de 110 animais relatada no workshop de 2025
como fonte P0 distinta, ainda não localizada.

**Motivo:** a auditoria do ZIP encontrou 904 imagens a 15 m, rótulos de detecção
e polígonos de cocho, mas nenhum peso ou `animal_id`. Isso coincide com o artigo
longitudinal de 2026 e difere das aproximadamente 10 mil amostras a 10 m
descritas no estudo de peso.

**Consequência SRP:** o adaptador UAV produz somente um inventário estrutural.
Ele não reutiliza nem contorna o importador de manifestos de regressão, cuja
responsabilidade exige vínculo confirmado entre imagem, animal, evento e peso.

## 2026-08-22 — Priorizar fontes brasileiras de Nelore

**Decisão:** o domínio final será construído prioritariamente com dados
brasileiros de Nelore. O dataset 3D da UNESP e a base UAV de peso descrita em
2025 são as fontes P0; o ZIP longitudinal público do
NelloreBeefCattleDataset é fonte auxiliar P1. CowDB permanece como baseline
técnico, não como representação do cenário final.

**Motivo:** raça, conformação corporal, manejo, câmera e ambiente alteram a
distribuição visual e limitam a transferência direta de bases Hereford/Angus.

**Condição:** somente fontes com correspondência confirmada entre `animal_id`,
imagem, evento e peso real podem supervisionar a regressão. Bases de
segmentação, identificação ou fotos genéricas permanecem auxiliares.

## 2026-08-22 — SRP como regra principal e obrigatória

**Decisão:** todo módulo, classe e função deve obedecer ao Single Responsibility
Principle. Componentes de composição podem orquestrar o fluxo, mas cada regra
específica deve permanecer em seu próprio componente testável.

**Motivo:** o projeto combinará aquisição, visão computacional, treinamento,
avaliação e produto de campo. Separar essas responsabilidades reduz acoplamento,
evita notebooks/arquivos monolíticos e permite evoluir cada etapa sem alterar as
demais.

**Critério de aceite:** mudanças com mais de um motivo independente para alterar
o mesmo componente devem ser separadas antes da aprovação, salvo exceção
temporária documentada neste arquivo.

## 2026-08-22 — Começar controlado e com RGB lateral

**Decisão:** o primeiro baseline usa uma única imagem lateral, um animal por
imagem e captura padronizada.

**Motivo:** permite validar rapidamente dados, regressão e avaliação. Fotos
livres, múltiplos animais e múltiplas vistas adicionam ambiguidades distintas.

## 2026-08-22 — Separar conjuntos por animal

**Decisão:** todas as imagens e eventos de um `animal_id` ficam na mesma
partição.

**Motivo:** impedir que o modelo reconheça o mesmo indivíduo no teste e produza
uma métrica excessivamente otimista.

## 2026-08-22 — Bases públicas validam o pipeline, não o domínio brasileiro

**Decisão:** CowDB e CowDatabase2 serão usadas para reprodução inicial; dados
locais serão necessários para o alvo final.

**Motivo:** raça, manejo, câmera e ambiente criam mudança de domínio.

## 2026-08-22 — R² não será métrica de decisão isolada

**Decisão:** priorizar MAE/MAPE e erros por faixa, mantendo R² como informação
complementar.

**Motivo:** grande amplitude de peso pode gerar R² alto mesmo com erro absoluto
operacionalmente inadequado.
