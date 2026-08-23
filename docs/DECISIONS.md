# Registro de decisões

## 2026-08-22 — Reservar calibração independente no split comercial

**Decisão:** a trilha comercial usa quatro partições agrupadas por animal:
60% treino, 15% validação, 10% calibração e 15% teste. O fluxo histórico de
pesquisa continua com seus três grupos originais para preservar reprodução.

**Motivo:** validação escolhe o modelo, enquanto calibração estima incerteza e
regras de rejeição depois dessa escolha. Misturar essas funções ou consultar o
teste produziria intervalos otimistas.

**Cadeia de entrada:** o split aceita somente manifesto selado acompanhado de
seu relatório. SHA-256, dHash, hash do manifesto e `snapshot_id` são verificados
novamente; qualquer alteração posterior à selagem bloqueia o processo.

**Consequência:** todas as visitas do mesmo animal permanecem no mesmo grupo.
O relatório registra contagens e distribuição de peso. A proporção é inicial e
não garante calibração útil se o piloto tiver poucos animais; tamanho e
representatividade serão avaliados antes de publicar intervalos.

## 2026-08-22 — Não calibrar incerteza com validação ou teste reutilizados

**Decisão:** não estimar intervalo conformal nem regra estatística de rejeição
do B2 usando as previsões de teste ou a validação que selecionou o checkpoint.

**Motivo:** o teste já mediu o resultado final e a validação já influenciou a
escolha da época. Reutilizá-los como calibração produziria cobertura otimista e
uma aparência indevida de segurança, principalmente com apenas 154 animais.

**Consequência:** o futuro snapshot próprio reservará animais para calibração
independente, separados de treino, validação e teste. Até lá, a rejeição cobre
somente qualidade técnica e domínio conhecido, sem intervalo probabilístico.

## 2026-08-22 — Selar a coleta antes da divisão e do treinamento

**Decisão:** toda coleta aprovada recebe manifesto canônico, SHA-256 de cada
arquivo, dHash perceptual, `snapshot_id` e relatório de proveniência antes de
criar splits ou treinar modelos.

**Motivo:** caminhos diferentes podem conter a mesma imagem e a ordem de um CSV
não deve mudar a identidade do conjunto. Pesos, políticas ou autorizações
alterados também precisam gerar proveniência diferente e auditável.

**Consequência:** duplicatas exatas bloqueiam a selagem; similaridade perceptual
gera alerta para revisão humana. Snapshots existentes nunca são sobrescritos.
Uma correção cria nova versão e preserva o histórico anterior.

## 2026-08-22 — Separar irreversivelmente B2 e candidato comercial

**Decisão:** classificar o B2 como modelo somente de pesquisa, com
`commercial_use_allowed: false` no descritor e na saída de inferência. O futuro
candidato comercial terá outro identificador, inicialização aleatória e dados
próprios ou expressamente licenciados.

**Motivo:** o B2 deriva de pesos ImageNet-1K e foi ajustado no CowDB; as
permissões comerciais necessárias não estão confirmadas. O código BSD da
torchvision não resolve sozinho os direitos dos pesos e dos dados.

**Consequência:** o B2 não será distribuído em produto, usado para destilação
nem para gerar pseudorrótulos comerciais. Uma promoção futura exige inventário
de proveniência, autorizações, validação em Nelore e revisão jurídica.

## 2026-08-22 — Exigir autorização rastreável na coleta piloto

**Decisão:** nenhuma imagem própria entrará na trilha comercial sem vínculo a
uma autorização vigente da mesma fazenda, permitindo treinamento de modelos e
uso comercial. A coleta será auditada por política versionada antes do treino.

**Motivo:** ser proprietário do código ou ter capturado a foto não documenta,
por si só, todo o escopo de uso acordado com a fazenda. O registro separado
permite bloquear autorizações ausentes, vencidas, revogadas ou insuficientes.

**Consequência:** documentos assinados ficam fora do Git em armazenamento
seguro; o manifesto guarda somente `authorization_id` e a referência controlada.
O auditor também verifica peso, horários, marcador, unicidade, imagem e
qualidade técnica, mas não substitui revisão jurídica ou inspeção de pose.

## 2026-08-22 — Bloquear inferência quando a qualidade técnica falhar

**Decisão:** aplicar antes da inferência uma política versionada que rejeita
imagens com resolução, proporção, exposição, saturação ou nitidez fora dos
limites definidos. Uma rejeição produz JSON com peso nulo e motivos explícitos;
o modelo não é carregado.

**Motivo:** fotos tecnicamente inadequadas podem produzir um número plausível,
mas sem sustentação. O gate reduz esse risco e permanece separado da inferência
para que seus limites possam ser calibrados sem retreinar a EfficientNet.

**Calibração inicial:** as métricas foram auditadas em 308 imagens laterais do
CowDB. Os limites são conservadores e destinam-se a capturas claramente ruins,
não a certificar conformidade zootécnica. As 308 imagens passaram pela política
inicial; imagens sintéticas pequenas, verticais, escuras e sem nitidez foram
rejeitadas. A suíte completa terminou com 87 testes aprovados.

**Consequência:** aprovação técnica não confirma presença de bovino, raça,
corpo inteiro, oclusão, distância nem vista lateral. Essas verificações exigem
detecção/segmentação e dados da futura coleta piloto de Nelore. A política
permanece heurística e experimental até essa calibração externa.

## 2026-08-22 — Empacotar o B2 como inferência experimental verificável

**Decisão:** disponibilizar o B2 como primeiro pacote de inferência local, com
descritor versionado, model card obrigatório, verificação SHA-256 do checkpoint
e saída JSON rastreável. O pacote permanece explicitamente experimental e não
está pronto para produção.

**Motivo:** o modelo é suficiente para validar a integração técnica, mas foi
treinado somente em 154 bovinos Hereford do CowDB e ainda não foi validado em
Nelore nem no ambiente brasileiro. Separar o pacote do futuro microsserviço
preserva a responsabilidade única e permite reutilizar a mesma inferência em
CLI, API e testes.

**Verificação:** sobre a imagem conhecida do animal `cowdb_009`, com peso real
de 416 kg, a execução em GPU retornou 426,3326 kg. O registro histórico é
426,3531 kg; a diferença de 0,0205 kg é compatível com variação numérica entre
ambientes de execução. Todos os 82 testes automatizados passaram.

**Consequência:** nenhuma previsão será executada se o checkpoint não tiver a
assinatura esperada ou se o model card estiver ausente. A saída inclui versão,
hash, dispositivo e limitações, e não deve ser usada em venda, dosagem,
tratamento ou substituição de balança aferida.

## 2026-08-22 — Adiar todos os pedidos externos de dados

**Decisão:** não enviar neste momento solicitações à UFGD/Embrapa sobre as
bases UAV de 2025 e 2026 nem à UNESP sobre a base Kinect/3D.

**Motivo:** decisão explícita do responsável pelo projeto. O adiamento controla
dependências externas sem alterar a prioridade técnica das fontes.

**Consequência:** nenhuma mensagem será preparada ou enviada automaticamente.
O trabalho segue nas tarefas locais independentes de novos dados: inferência,
qualidade de captura, pacote de coleta piloto e documentação do modelo atual.

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
