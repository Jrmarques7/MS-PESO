# Registro de decisões

## 2026-08-22 — Rejeitar fusão da altura PLY no gate de validação

**Decisão:** preservar a leitura e a auditoria de nuvens PLY, mas não treinar
uma variante RGB + geometria de uma ou três vistas com a segmentação atual. O
teste permanece intocado.

**Motivo:** a altura robusta teve correlação com peso de +0,437 no treino e
+0,402 na validação, porém sua regressão linear obteve MAE de 46,89 kg na
validação, contra 37,63 kg do B2. Mais importante, a correlação da altura com o
resíduo de validação do B2 foi -0,016: ela não acrescenta sinal complementar.
A correlação de apenas +0,36 com as alturas manuais também mostra que piso e
partes do cenário ainda limitam a medida física extraída.

As vistas direita e superior confirmaram a decisão. A correlação com peso na
validação foi somente +0,149 e +0,258, respectivamente; seus MAEs lineares
foram 50,96 e 49,44 kg. A correlação com o resíduo do B2 foi -0,106 na direita
e +0,210 na superior, insuficiente para justificar uma fusão de maior custo.

**Consequência:** não consultar o teste nem gastar GPU nessa fusão. Uma nova
tentativa 3D exigirá segmentação corporal realmente tridimensional, combinação
das três câmeras ou reconstrução fornecida pelos autores; somente então passará
novamente pelo mesmo gate de treino/validação.

## 2026-08-22 — Não promover a fusão lateral + superior

**Decisão:** manter B2 como baseline oficial e preservar A5 apenas como suporte
multivista reproduzível. Não repetir esta configuração nas seeds 43 e 44.

**Motivo:** no mesmo teste de 20 animais, A5 piorou o MAE de 34,21 para 37,91
kg, o MAPE de 9,39% para 10,32% e o viés de +7,07 para +18,54 kg. No bootstrap
pareado, a chance de MAE menor foi 18,8%. A piora de viés de +11,47 kg teve
IC95% de +2,57 a +19,38 kg, inteiramente acima de zero.

**Consequência:** uma vista adicional não será presumida útil por si só. O
próximo avanço deve priorizar escala/geometria calibrada ou dados do domínio
brasileiro, evitando selecionar novas regras sobre o mesmo teste pequeno.

## 2026-08-22 — Não promover recorte nem máscara guiados por profundidade

**Decisão:** manter B2 como baseline oficial. Preservar o preparador de caixas
para pesquisa, mas não promover A1 ou A2.

**Motivo:** A1 reduziu o viés, porém piorou MAE, RMSE e MAPE na seed 42. A2
teve excelente resultado pontual na seed 42, mas suas três execuções variaram
de 28,16 a 45,00 kg de MAE. A média de 36,97 ± 8,44 kg ficou pior e muito mais
instável que os 32,10 ± 2,43 kg do B2.

**Consequência:** não continuar refinando limiares sobre o mesmo teste. A
próxima ablação deve adicionar informação independente — outra vista ou
geometria explicitamente segmentada — sem usar o teste para selecionar regras
de pré-processamento.

## 2026-08-22 — Não fundir a cena de profundidade bruta ao RGB

**Decisão:** rejeitar o A6 de fusão global RGB + profundidade e manter o B2
RGB como baseline oficial. Não repetir esta configuração em outras seeds.

**Motivo:** no mesmo teste de 20 animais, o A6 piorou o MAE em 9,06 kg e o
RMSE em 8,50 kg; ambos os intervalos de confiança pareados ficaram inteiramente
acima de zero. A profundidade contém a silhueta, mas também fundo, estruturas do
curral e muitos pixels inválidos. A representação aprendida com apenas 109
animais acrescentou ruído em vez de geometria útil.

**Consequência:** a profundidade será tratada como informação geométrica para
recorte, máscara ou medidas após remoção do fundo. O carregamento multimodal
permanece no projeto para experimentos reproduzíveis, mas não será promovido ao
fluxo principal.

## 2026-08-22 — Manter EfficientNet-B0 após avaliar ConvNeXt-Tiny

**Decisão:** não promover o B3 ConvNeXt-Tiny. O B2 EfficientNet-B0 com
amostragem uniforme permanece o baseline visual oficial.

**Motivo:** em três seeds, o B3 reduziu o MAE médio em apenas 0,46 kg, mas
piorou RMSE em 6,25 kg, MAPE em 0,44 ponto percentual, viés em 7,00 kg e R²
em 0,12. Na comparação pareada da seed 42, o intervalo de confiança da
diferença de MAE cruzou zero. Os maiores erros continuaram concentrados nos
dois animais abaixo de 350 kg e foram ainda maiores que no B2.

**Consequência:** não gastar novas execuções apenas refinando a arquitetura no
CowDB pequeno. O próximo ganho deve vir de informação adicional ou controle da
imagem — recorte/segmentação, escala, múltiplas vistas ou profundidade — e de
dados brasileiros com pesagem vinculada.

## 2026-08-22 — Priorizar a fonte UAV 3D de 2026 e pedir esclarecimento sobre 2025

**Decisão:** registrar o estudo Embrapa/UFGD de nuvem de pontos de 2026 como
fonte P0 para solicitação. Manter a base de peso anunciada em 2025 separada,
mas alterar seu estado para "publicação não reproduzida", pois os endereços
públicos encontrados levam ao ZIP longitudinal sem pesos já auditado.

**Motivo:** o estudo de 2026 confirma imagens capturadas no mesmo dia da
pesagem, identificação numérica individual e regressão por volume 3D. Contudo,
usa apenas sete animais para calibração, não informa claramente o total final
retido após exclusões e não disponibiliza os dados. Seu RMSE de 8,35 kg não é
diretamente comparável ao teste RGB lateral do MS-PESO.

**Consequência:** não haverá tentativa de fabricar um manifesto de regressão a
partir do ZIP longitudinal. A aquisição de dados agora inclui duas perguntas
distintas aos autores: acesso ao material 3D de 2026 e correção/localização da
base de aproximadamente 10 mil amostras anunciada em 2025.

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
