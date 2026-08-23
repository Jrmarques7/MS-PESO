# Fontes brasileiras de dados bovinos

Revisão: 2026-08-22.

Este documento separa fontes capazes de supervisionar a regressão de peso de
fontes auxiliares para segmentação, detecção, identificação e diversidade
visual. Uma imagem de Nelore sem peso associado não pode ser usada como exemplo
supervisionado de regressão.

## Prioridade operacional

| Prioridade | Fonte | Uso principal | Peso real associado | Acesso |
|---|---|---|---|---|
| P0 | UNESP Kinect/3D | regressão de peso Nelore | sim | solicitar aos autores |
| P0 | Nuvem de pontos UAV de 2026 | regressão 3D de Nelore | sim | solicitar dados e protocolo aos autores |
| P0 | Estudo de peso UAV de 2025 | regressão dorsal de Nelore | relatado; publicação dos arquivos não reproduzida | esclarecer com autores |
| P1 | NelloreBeefCattleDataset longitudinal | detecção e morfometria UAV | não no ZIP público | download público |
| P1 | Nelore Instance Segmentation | segmentação anatômica | não identificado | público, CC BY 4.0 |
| P2 | Embrapa facial | identificação individual | não | disponibilidade a confirmar/solicitar |
| P3 | Wikimedia Commons | detecção e diversidade visual | não | público, licença por arquivo |

Todos os pedidos formais de dados foram adiados por decisão do responsável pelo
projeto: UNESP Kinect/3D, base UAV de peso de 2025 e nuvem de pontos UAV de
2026. O trabalho local continua com as fontes já disponíveis e com a preparação
do pipeline; o adiamento não deve ser interpretado como descarte das fontes.

Elas não devem ser concatenadas diretamente: UAV RGB dorsal e Kinect 3D têm
geometria, resolução e protocolo distintos. Cada modalidade terá baseline e
avaliação próprios; fusão ou transferência entre elas será um experimento
posterior.

## 1. NelloreBeefCattleDataset — UFGD/Embrapa

- Repositório: <https://github.com/EvertonTetila/NelloreBeefCattleDataset>
- Publicação Embrapa sobre peso: <https://www.alice.cnptia.embrapa.br/alice/handle/doc/1185904>
- Publicação longitudinal de 2026: <https://doi.org/10.1016/j.compag.2026.111559>
- Raça/domínio: Nelore em confinamento, Mato Grosso do Sul, Brasil.
- Modalidade: imagens RGB dorsais obtidas por UAV.
- Estado: ZIP público auditado em 2026-08-22.

O material do workshop de 2025 relata aproximadamente 10 mil imagens/amostras
de 110 animais individualmente identificados, treze voos, uso de balança como
referência e finalidade de estimativa de peso. O artigo de 2026 descreve 904
imagens ao longo de 112 dias, das quais 370 foram anotadas para detecção, e
enfatiza acompanhamento morfológico populacional, não estimativa direta de
peso.

Há ainda diferença de protocolo: o workshop menciona voos a 10 m, enquanto o
README atual e o artigo longitudinal mencionam 15 m e doze operações. A
auditoria confirmou que o ZIP atualmente ligado pelo GitHub corresponde à
versão longitudinal de 2026, não à base de peso descrita no workshop.

### Resultado da auditoria do ZIP público

- tamanho: 9.683.307.424 bytes;
- estrutura: 1.647 arquivos e 69 diretórios;
- imagens UAV brutas: 904, em doze sessões entre 2024-07-10 e 2024-10-27,
  sempre no diretório `15m`;
- corpus de detecção: 232 imagens, com 232 JSON LabelMe e 232 rótulos YOLO;
- divisão de detecção: 97 imagens de treino, 47 de validação e 88 de teste;
- classes de detecção: `cattle-back` e `cattle-head`;
- corpus de cocho: 23 imagens e 23 polígonos LabelMe;
- nenhum CSV, XLS/XLSX, Parquet ou outro arquivo de metadados de peso;
- nenhum mapeamento persistente de `animal_id` para indivíduo, imagem e data;
- nenhum arquivo de licença identificado no ZIP.

O total de 232 imagens em `annotations_v2` também difere das 370 anotações
relatadas no artigo. O inventário registra o conteúdo efetivamente publicado no
ZIP atual; os 138 itens restantes não devem ser presumidos disponíveis.

O README interno confirma que as anotações se destinam à detecção de gado e à
segmentação do cocho. Os JSON podem conter vários animais na mesma imagem, mas
as caixas representam partes (`cattle-back` e `cattle-head`), não identidades.

**Conclusão:** o ZIP público não pode supervisionar regressão de peso. Ele é
válido para detecção, exclusão do fundo/cocho, estudo de morfologia populacional
e pré-treinamento no domínio visual de Nelore visto por UAV.

O inventariador reproduzível está em
`src/ms_peso/importers/nellore_uav.py`. Depois da extração em armazenamento
externo, execute:

```bash
python -m ms_peso.inspect_nellore_uav \
  --dataset-root /caminho/NelloreBeefCattleDataset \
  --output artifacts/nellore_uav_inventory.json
```

O estudo de peso de 2025 permanece como uma fonte P0 separada, mas a busca
deixou de ser apenas "arquivo não localizado". A página pública de datasets do
autor e a notícia institucional da UFGD apontam para o mesmo repositório GitHub
já auditado, cujo ZIP contém 904 imagens e nenhum peso. Assim, a afirmação de
que cerca de 10 mil amostras foram disponibilizadas publicamente não pôde ser
reproduzida. O próximo passo é pedir aos autores os arquivos corretos ou uma
correção do endereço; não devemos presumir que os dados estejam ocultos no ZIP.

## 2. Embrapa/UFGD — nuvem de pontos UAV com peso real (2026)

- Registro Embrapa:
  <https://www.alice.cnptia.embrapa.br/alice/handle/doc/1187153>
- Texto completo: <https://www.scitepress.org/publishedPapers/2026/149255/pdf/index.html>
- DOI: <https://doi.org/10.5220/0014925500004018>
- Raça/domínio: 70 bovinos Nelore presentes em um confinamento na Fazenda
  Campanário, Laguna Carapã-MS.
- Aquisição: aproximadamente 190 imagens nadirais com DJI Phantom 4 Advanced,
  a 10 m, no mesmo dia da pesagem individual.
- Modalidade: nuvem de pontos densa reconstruída por Structure from Motion
  (SfM), segmentação DBSCAN e volume baseado em voxels.

Os animais tinham etiquetas numéricas únicas, e a correspondência entre
observações aéreas e pesos foi feita manualmente por inspeção visual. O modelo
linear foi calibrado com apenas sete animais em pé. O artigo relata RMSE de
6,03 kg nesse pequeno subconjunto e RMSE de 8,35 kg nos animais restantes,
além de erro relativo médio de 2,02% no corpo do texto (o resumo informa
aproximadamente 2,29%).

Esse resultado é promissor, mas ainda não é uma referência diretamente
comparável ao MS-PESO: o artigo não informa claramente quantos dos 70 animais
passaram pelos filtros finais, aceita somente animais em pé com segmentação
correta, exigiu refinamento manual de sete segmentos e usa regressão sobre
volume 3D, enquanto nosso teste atual usa 20 imagens RGB laterais de indivíduos
Hereford. O número 8,35 kg não deve ser apresentado como uma meta equivalente
sem reproduzir a avaliação no mesmo protocolo.

Não foi localizada uma seção de disponibilidade dos dados nem um download dos
arquivos. A licença CC BY-NC-ND 4.0 identificada no artigo cobre a publicação,
não concede automaticamente uso do dataset. Esta é uma fonte P0 concreta para
solicitação: imagens originais, nuvem densa, segmentos por animal, etiquetas,
pesos, lista de exclusões e partição exata de calibração/validação.

## 3. UNESP — Kinect/3D com peso real

- Artigo: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10215216/>
- DOI: <https://doi.org/10.3390/ani13101679>
- Trabalho no repositório UNESP:
  <https://repositorio.unesp.br/bitstreams/248a6fbe-d5d0-42b9-82e8-97dbb48f956c/download>
- Contato do autor correspondente: `otavio.machado@unesp.br`.

O método descreve 450 bovinos machos Nelore distribuídos em quatro
experimentos, peso corporal de 359 a 665 kg e pesagem em balança eletrônica com
precisão de 0,5 kg. O Kinect 1473 foi instalado sobre o corredor; foram obtidos
dez frames RGB/infravermelho por animal durante 15 segundos, com seleção manual
do melhor frame para a análise publicada.

O resumo do artigo menciona 1.350 observações, mas a seção de métodos, as
tabelas e a dissertação descrevem 450 animais/imagens únicas. Para o planejamento
do MS-PESO, adotamos **450 animais únicos** e trataremos qualquer número maior
como registros ou reorganizações experimentais até confirmação dos autores.

Os dados não têm download público indicado. O artigo afirma que estão
disponíveis mediante solicitação ao autor correspondente. A licença CC BY 4.0
do artigo não concede automaticamente licença sobre os arquivos do dataset;
os termos devem constar da resposta dos autores.

### Conteúdo a solicitar

- frames RGB e profundidade originais ou processados;
- `animal_id`, experimento e data;
- peso vivo e, se permitido, peso de carcaça;
- medidas/features extraídas e máscaras;
- regra de seleção do melhor frame;
- termos de uso, publicação e compartilhamento de modelos derivados.

## 4. Nelore Instance Segmentation — Roboflow Universe

- Fonte:
  <https://universe.roboflow.com/henriques-workspace-lhcrk/nelore-instance-segmentation>
- 323 imagens na versão verificada;
- tarefa: segmentação de instâncias;
- licença declarada: CC BY 4.0;
- classes: `head`, `hump`, `neck`, `posterior`, `rump`, `top_line`.

Não foi identificado peso associado. A utilidade é treinar ou avaliar
segmentação anatômica e estudar se regiões como cupim, garupa e linha dorsal
ajudam a regressão. Não deve entrar diretamente como linha de treino com
`weight_kg`.

Antes de incorporar, registrar versão exportada, autoria/atribuição exigida e
verificar se todas as imagens de origem são compatíveis com a licença declarada.

## 5. Embrapa — reconhecimento facial de Nelore

- Artigo:
  <https://www.mdpi.com/2624-7402/6/3/169>
- Registro técnico Embrapa:
  <https://ainfo.cnptia.embrapa.br/digital/bitstream/doc/1169934/1/Reconhecimento-facial-de-bovinos-da-raca-Nelore-por-meio-de-visao.pdf>
- 2.210 frames de 47 animais Nelore: 20 machos e 27 fêmeas;
- imagens faciais associadas ao identificador eletrônico individual;
- captura com GoPro 5 ao lado do brete, em instalações da Embrapa Gado de
  Corte, Campo Grande-MS.

Não é uma base de peso. Sua contribuição potencial é reconhecimento de
`animal_id`, deduplicação e ligação automática entre uma imagem corporal e o
histórico de pesagens. A disponibilidade dos arquivos e os termos de uso devem
ser confirmados com os autores antes de planejar integração.

## 6. Wikimedia Commons — categoria Nelore

- Fonte: <https://commons.wikimedia.org/wiki/Category:Nelore>
- 43 arquivos na revisão realizada.

Pode fornecer diversidade de aparência, pose e fundo para detecção ou análise
exploratória. Não contém peso associado e a categoria não possui uma única
licença para todas as imagens: cada página de arquivo deve ser verificada e a
atribuição correspondente preservada.

Não é prioridade para o baseline de peso e não deve ser usada para inflar o
número de observações supervisionadas.

## Esquema brasileiro desejado

O contrato mínimo continua sendo:

```text
animal_id | event_id | image_path | weight_kg | captured_at
```

Para capturas brasileiras, buscar também:

```text
breed | sex | age_months | farm_id | lot_id | view | camera_id
camera_height_m | distance_m | scale_marker | body_condition_score
```

Em dados longitudinais, todas as imagens do mesmo `animal_id` permanecem no
mesmo split, mesmo quando os pesos e as datas forem diferentes.

## Próximas ações

1. Manter os três pedidos P0 adiados até decisão explícita do responsável pelo
   projeto; não preparar nem enviar mensagens automaticamente.
2. Executar o inventariador se o ZIP público longitudinal for copiado para um
   armazenamento com espaço suficiente e extraído.
3. Manter as fontes auxiliares fora da regressão até existir uma estratégia
   experimental específica para cada fonte.
4. Avançar localmente no fluxo de inferência, controle de qualidade da captura
   e preparação do piloto com pesagem vinculada.
