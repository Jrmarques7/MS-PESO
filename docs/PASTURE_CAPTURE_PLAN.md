# Plano de captura lateral no pasto

Revisão: 2026-08-23.

## 1. Objetivo

Formar uma base própria de bovinos no cenário real do produto: foto ou vídeo
lateral no pasto, ligado ao peso de balança do mesmo animal e evento. A coleta
deve permitir treinar, calibrar e testar um futuro modelo comercial sem usar o
B2/CowDB como verdade de referência.

A unidade estatística é o **animal**, e a unidade de captura é o **evento**.
Vinte quadros do mesmo vídeo continuam sendo uma única observação independente.

## 2. Metas progressivas de animais únicos

As quantidades abaixo são metas operacionais, não garantias de desempenho. A
decisão de ampliar a coleta será feita por curva de aprendizado, erro por grupo
e intervalo de confiança.

| Fase | Animais únicos | Finalidade | Pode liberar produto? |
|---|---:|---|---|
| Ensaio de campo | 10–20 | corrigir câmera, planilha, identidade e logística | não |
| Piloto diagnóstico | 50 | medir domínio, distribuição de pesos e falhas de captura | não |
| Primeiro treinamento próprio | 150–200 | treinar e comparar um candidato inicial | não |
| Robustez multissítio | 300–500+ | incluir outras datas, condições e fazendas | somente após todos os gates |

Quando houver mais de uma fazenda, reservar como teste externo pelo menos uma
fazenda ou período nunca usado em treino, validação ou calibração. O número
final não será escolhido pelo total de vídeos, mas pelo total de animais e pela
representatividade dos grupos.

Como referência de escala, trabalhos publicados variam muito: há uma base de
72 bovinos livres no pasto, estudos RGB com 107 animais e múltiplas imagens, e
o estudo 3D da UNESP com 450 Nelore. Esses números demonstram viabilidade, mas
não substituem validação no nosso domínio.

## 3. Peso: coletar primeiro, estratificar depois

Não excluir animais por peso durante o ensaio inicial. Registrar todos os pesos
reais disponíveis. Depois dos primeiros 30–50 animais:

1. calcular mínimo, máximo, mediana e quartis da população observada;
2. contar animais únicos por quartil, sexo, raça/cruzamento, lote e fazenda;
3. identificar caudas e grupos pouco representados;
4. direcionar as próximas capturas para os vazios reais;
5. manter o teste com a distribuição natural do uso, sem balanceá-lo
   artificialmente.

Não usar a estimativa do B2 para preencher `weight_kg`. O rótulo vem somente de
balança identificada. Se o peso não estiver disponível, preservar o vídeo como
material auxiliar e deixar o peso vazio; ele não entra na regressão.

## 4. Rotina por animal e evento

1. Confirmar `animal_id`, fazenda, lote, raça/cruzamento e sexo.
2. Fazer uma foto aproximada do identificador, separada do vídeo corporal.
3. Registrar o peso em balança aferida e o horário exato.
4. Em até 30 minutos antes ou depois da pesagem, gravar o vídeo lateral no
   pasto ou área imediatamente adjacente ao manejo.
5. Gravar preferencialmente 4–8 segundos; vídeos de 2 segundos podem ser
   aceitos quando fornecerem ao menos três quadros bons e distintos.
6. Manter o celular horizontal, sem zoom digital e aproximadamente
   perpendicular ao tronco.
7. Manter cabeça, tronco, quartos e patas inteiros dentro do quadro.
8. Buscar um animal principal; registrar outros bovinos, cercas e oclusões em
   vez de esconder a dificuldade.
9. Se possível, fazer duas tomadas curtas do mesmo lado. Elas pertencem ao
   mesmo evento e não contam como novos animais.
10. Preservar o arquivo original; seleção, recorte e compressão geram derivados.

O lado preferencial do primeiro candidato é `left`. Uma captura direita útil
deve ser preservada e rotulada como `right`, nunca renomeada como esquerda.

## 5. Configuração inicial de captura

- resolução preferencial: 1080p; 720p é aceitável para ensaio;
- taxa preferencial: 30 fps;
- orientação: horizontal;
- distância: suficiente para incluir o corpo inteiro, sem zoom digital;
- duração-alvo: 4–8 s; máximo do Vídeo V1: 10 s;
- movimento: operador parado ou deslocamento lento e contínuo;
- luz: evitar contraluz extremo; não excluir automaticamente sol e sombra;
- escala: marcador conhecido no plano do animal quando for viável;
- áudio e dados pessoais: não são necessários para o modelo.

Distância, dispositivo e resolução devem permanecer registradas mesmo quando
variarem. O modelo final precisa conhecer a variação real, mas o ensaio inicial
deve reduzir mudanças desnecessárias para revelar os problemas principais.

## 6. O que aceitar, rejeitar e preservar

### Aceitar para seleção de quadros

- um animal principal claramente identificável;
- corpo inteiro e pose aproximadamente lateral;
- foco e exposição suficientes;
- identidade e peso vinculados ao evento;
- autorização de treinamento e uso comercial confirmada.

### Marcar como rejeitado para regressão

- identidade incerta ou peso de outro evento;
- cabeça, tronco, quartos ou patas cortados de forma importante;
- animal principal encoberto por outro bovino;
- pose frontal/traseira dominante;
- movimento ou desfoque durante todo o vídeo;
- arquivo editado sem preservação do original.

Vídeos rejeitados não devem ser apagados. Com autorização, eles podem ajudar a
treinar futuramente detecção de falhas, segmentação e rejeição automática.

## 7. Metadados mínimos

O template `data/templates/pasture_video_manifest.csv` registra:

- arquivo, animal e evento;
- peso, horários, câmera e balança;
- raça, sexo, fazenda e lote;
- lado, duração e distância aproximada;
- quantidade visível de animais, corpo inteiro, pose e oclusão;
- marcador de escala, autorização, permissão comercial, qualidade e observações.

Depois que o Vídeo V1 selecionar os quadros, cada frame destinado ao treino deve
ser materializado no manifesto comercial de imagens. Todos os frames do mesmo
animal permanecem no mesmo split.

## 8. Estrutura de armazenamento

```text
data/raw/pasture/
  farm_001/
    animal_0001/
      event_20260823_001/
        identity.jpg
        lateral_left_01.mp4
        lateral_left_02.mp4
data/interim/
  pasture/
    batch_20260823_001/
      batch_metadata.json
      pasture_video_manifest.csv
      authorization_registry.csv
      selection_v001/
        pilot_manifest.csv
        selection_report.json
        selected_frames/
```

Os vídeos não entram no Git. Use backup e conteúdo imutável; correções são
novas versões do manifesto.

## 9. Fluxo operacional implementado

Crie um lote novo, que nunca substitui um lote anterior:

```powershell
python -m ms_peso.init_collection_batch `
  --batch-id batch_20260823_001 `
  --farm-id farm_001
```

Depois de preencher o inventário e revisar manualmente os vídeos, selecione um
quadro por evento/vista:

```powershell
python -m ms_peso.select_collection_frames `
  --manifest data/interim/pasture/batch_20260823_001/pasture_video_manifest.csv `
  --video-root data/raw/pasture `
  --image-root data `
  --output-directory data/interim/pasture/batch_20260823_001/selection_v001
```

Esse comando não usa o B2 nem qualquer outro estimador de peso. O peso é
copiado exclusivamente do registro da balança. A seleção técnica mede somente
resolução, exposição e nitidez; por isso, todos os quadros saem como `review`
e precisam de uma segunda inspeção humana antes da auditoria comercial.

## 10. Checklist antes de encerrar o dia

- [ ] todo vídeo possui `animal_id` e `event_id` únicos e legíveis;
- [ ] peso e horário foram copiados da balança, não estimados;
- [ ] relógios de captura e pesagem estão sincronizados;
- [ ] arquivos abrem e têm backup;
- [ ] autorização corresponde à fazenda e ao período;
- [ ] rejeições e anomalias foram registradas;
- [ ] nenhum arquivo original foi recortado ou sobrescrito;
- [ ] contagem de animais é separada da contagem de vídeos e quadros.

## 11. Gates antes de treinar

1. auditar direitos e metadados;
2. verificar arquivos e identidade;
3. analisar distribuição real dos pesos;
4. revisar visualmente pose, corpo inteiro e oclusão;
5. deduplicar e selar a coleta;
6. separar por `animal_id` em treino, validação, calibração e teste;
7. reservar teste externo quando houver outra fazenda ou período;
8. treinar do zero na trilha comercial;
9. abrir o teste somente depois de congelar os critérios;
10. manter `commercial_use_allowed: false` até revisões técnica, jurídica e
    operacional.

## 12. Papel de bases externas

A UNESP pode fornecer um sinal inicial de morfologia e peso de Nelore se os
arquivos e direitos comerciais forem concedidos. Como foi capturada com Kinect
3D em corredor controlado, ela não substitui o teste nem a adaptação em vídeo
lateral de pasto. Bases externas e próprias devem manter proveniência, modelos
e relatórios separados até uma comparação protegida demonstrar benefício.

Referências de planejamento:

- [UNESP — imagens 3D e peso de Nelore](https://pmc.ncbi.nlm.nih.gov/articles/PMC10215216/)
- [Base lateral de 72 bovinos livres no pasto](https://doi.org/10.1016/j.dib.2024.110835)
- [RGB lateral/superior com 107 bovinos](https://doi.org/10.3390/ani16162532)
