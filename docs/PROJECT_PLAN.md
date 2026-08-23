# Plano do projeto

> Regra arquitetural obrigatória: todo desenvolvimento deve respeitar o
> **Single Responsibility Principle (SRP)** conforme
> [Princípios de engenharia](ENGINEERING_PRINCIPLES.md).

## 1. Problema

Estimar o peso vivo de bovinos sem contato e sem depender de uma pesagem
frequente, usando visão computacional. A balança continua sendo necessária na
fase de coleta e validação: ela fornece o valor de referência que ensina e mede
o modelo.

## 2. Usuário e uso inicial

O usuário-alvo inicial é um técnico ou produtor que grava um vídeo lateral
curto do animal no pasto ou em um ponto natural de passagem, sem precisar
conduzi-lo até uma balança. O sistema selecionará quadros válidos e estimará o
peso para acompanhar a evolução do lote. A captura precisa continuar guiada:
um animal por vez, corpo inteiro visível, pose lateral e distância ou referência
de escala controlada.

O primeiro marco científico continua recebendo uma imagem por inferência. A
captura por vídeo será construída sobre esse núcleo depois que segmentação,
pose e qualidade estiverem validadas. A balança permanece obrigatória durante a
coleta e validação dos dados, mas não na rotina posterior de estimativa. O
resultado não deve ser usado como única fonte para venda, dosagem veterinária
ou decisões em que um erro de peso possa causar dano.

## 3. Escopo do MVP

### Incluído

- bovinos de corte de uma população-alvo definida;
- uma imagem RGB lateral por evento;
- animal individual e corpo inteiro visível;
- regressão direta para peso vivo em kg;
- treinamento por transferência de aprendizado;
- estimativa acompanhada de indicadores de qualidade;
- avaliação com indivíduos nunca vistos durante o treino.

### Fora do primeiro marco

- vários animais na mesma imagem;
- fotografia aérea por drone;
- qualquer raça, idade e sistema produtivo sem recalibração;
- operação totalmente livre quanto a ângulo e distância;
- substituição certificada de balanças comerciais;
- aplicativo móvel de produção.

## 4. Hipóteses a testar

1. Uma ResNet18 pré-treinada supera o preditor ingênuo da média do treino.
2. Segmentar o animal reduz dependência do fundo e melhora o teste externo.
3. Referência física/câmera fixa reduz o erro em relação a fotos sem escala.
4. Duas vistas contêm informação complementar e reduzem o erro.
5. O desempenho cai ao mudar fazenda, raça ou câmera; ajuste local recupera
   parte dessa diferença.
6. A mediana de três a cinco bons quadros laterais de um vídeo é mais estável
   que a estimativa de um único quadro escolhido sem consenso.

## 5. Critérios de sucesso

Os limites finais devem ser definidos com o usuário de campo. Para o piloto:

- nenhum `animal_id` compartilhado entre treino, validação e teste;
- calibração independente sem `animal_id` compartilhado com os demais grupos;
- MAE e RMSE melhores que o baseline da média por margem relevante;
- MAPE global inicial abaixo de 10%;
- relatório de erro por faixa de peso, sexo, raça e origem;
- teste externo por fazenda ou por período antes de declarar uso real;
- toda predição rastreável até versão do modelo e protocolo de captura.
- seleção dos quadros feita sem consultar o peso real do evento;
- limiar de divergência entre quadros definido antes de abrir o teste final;
- resposta de vídeo rastreável até os instantes e notas dos quadros escolhidos.

R² será relatado, mas nunca utilizado sozinho como critério de sucesso.

## 6. Marcos

### M0 — Fundação

- documentação, esquema do manifesto e código-base;
- testes de integridade e separação por animal;
- configuração reproduzível.

### M1 — Reprodução pública

- manter registrado o ZIP longitudinal auditado como fonte auxiliar, sem peso;
- manter adiados, por decisão do responsável, os pedidos de dados UAV/SfM de
  2026, UAV de peso de 2025 e Kinect/3D da UNESP;
- manter CowDB como validação técnica enquanto os dados Nelore são preparados;
- manter experimentos separados por modalidade: lateral RGB, dorsal UAV e 3D;
- treinar média, regressão visual ResNet18 e EfficientNet-B0;
- publicar relatório com intervalos de confiança.

### M2 — Piloto local

- escolher raça, sexo e faixa de peso;
- coletar de 100 a 200 animais, preferencialmente em mais de uma visita;
- registrar balança, câmera e condições;
- auditar direitos, deduplicar e congelar snapshot antes da divisão;
- separar treino, validação, calibração e teste por animal;
- calibrar intervalos por animal sem reutilizar validação ou teste;
- pré-registrar critérios e consumir o teste final uma única vez;
- empacotar inferência interna com qualidade e intervalo, sem criar a API;
- comparar modelo público, ajuste fino e modelo local.

### M3 — Robustez

- segmentação do animal;
- detecção automática de pose/foto inválida;
- receber vídeo lateral curto e amostrar aproximadamente 15 a 30 quadros;
- eliminar quadros borrados, cortados, ocluídos, sem corpo inteiro ou com mais
  de um animal dominante;
- ranquear qualidade e selecionar de três a cinco quadros bons, distintos no
  tempo, sem usar o peso real na seleção;
- estimar cada quadro escolhido e agregar o resultado pela mediana;
- rejeitar o vídeo quando houver poucos quadros válidos ou divergência excessiva
  entre as estimativas;
- comparar o vídeo agregado com o melhor quadro único em avaliação separada por
  animal e evento;
- segunda vista ou profundidade;
- teste em outra fazenda, lote ou período.

### M4 — Produto

- adaptador HTTP de inferência criado e bloqueado até a promoção do modelo;
- endpoint de vídeo separado do endpoint de imagem, preservando o núcleo de
  inferência por quadro;
- retorno auditável com instantes selecionados, notas de qualidade, estimativas
  por quadro, resultado agregado e motivo de eventual rejeição;
- remoção do vídeo temporário ao final da requisição, salvo consentimento e
  política explícitos para formar uma coleta autorizada;
- interface de captura guiada na aplicação consumidora;
- estimativa de incerteza e regra de rejeição;
- monitoramento de deriva e rotina de recalibração;
- validação operacional e análise de custos.

## 7. Riscos principais e mitigação

| Risco | Consequência | Mitigação |
|---|---|---|
| Frames do mesmo animal em treino e teste | Métrica artificialmente alta | Split obrigatório por `animal_id` |
| Fotos distantes da pesagem | Rótulo incorreto | Captura e balança no mesmo evento |
| Fundo correlacionado com lote/peso | Atalho visual | Segmentação e teste em outro ambiente |
| Poucos indivíduos, muitas fotos | Falsa impressão de escala | Contabilizar animais e eventos, não frames |
| Mudança de raça/fazenda | Queda de generalização | Teste externo e ajuste local |
| Perspectiva sem referência | Tamanho físico ambíguo | Câmera fixa, calibração ou RGB-D |
| Muitos frames quase iguais | Confiança artificial no consenso | Seleção com diversidade temporal e avaliação por evento |
| Vídeo com quadros de animais diferentes | Peso agregado sem significado | Rastreamento do indivíduo e rejeição de troca/oclusão |
| Seleção de quadro ajustada pelo peso real | Vazamento e resultado otimista | Ranqueamento baseado apenas em imagem e protocolo congelado |
| Divergência entre quadros | Estimativa instável | Limiar calibrado e rejeição do vídeo |
| Distribuição de peso desigual | Viés nas faixas raras | Amostragem/ponderação apenas no treino |
| Uso fora do domínio | Predição perigosa | Incerteza, detecção de qualidade e rejeição |

## 8. Recursos necessários

- acesso aos animais e identificação individual;
- balança aferida e dados de peso sincronizados;
- celular capaz de gravar vídeo lateral; marcador ou medição de distância
  recomendados;
- autorização da propriedade e política para imagens/metadados;
- armazenamento versionado fora do Git para imagens;
- Google Colab com GPU ou máquina CUDA;
- responsável técnico para definir erro aceitável no uso real.

## 9. Regra de implementação

O SRP é critério de aceite para toda mudança. Coleta, contrato dos dados,
transformação, modelo, treinamento, métricas, inferência e interface são
responsabilidades distintas. Scripts de entrada podem coordená-las, sem
incorporar suas implementações. Código reutilizável não deve existir apenas em
notebooks.

## 10. Perguntas ainda abertas

- Qual raça ou cruzamento será atendido primeiro?
- Qual faixa de peso, sexo e fase produtiva?
- Há acesso a balança e identificação individual?
- Qual celular/câmera será o dispositivo inicial e como obteremos distância ou
  referência de escala no pasto?
- Quantos animais formarão uma amostra representativa de cada lote por evento?
- Quantas fazendas e quantas datas de coleta são possíveis?
- Qual erro em kg/% ainda produz uma decisão útil?
