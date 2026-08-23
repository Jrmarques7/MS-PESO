# Model card — candidato comercial MS-PESO

## Estado

Este arquivo é um template para um candidato futuro. Nenhum modelo comercial
real foi treinado, calibrado ou aprovado. O descritor correspondente permanece
`candidate_unapproved`, `production_ready: false` e
`commercial_use_allowed: false`.

## Origem obrigatória

- arquitetura inicializada aleatoriamente, sem ImageNet, B2 ou destilação;
- imagens próprias ou expressamente licenciadas para treinamento e uso comercial;
- autorizações, fazendas, animais, eventos, câmeras e balanças rastreáveis;
- snapshot selado e quatro partições sem compartilhamento de `animal_id`;
- checkpoint, calibração e avaliação ligados por SHA-256.

## Saída planejada

Para uma imagem tecnicamente aceita, o pacote retorna uma estimativa em kg e
um intervalo conformal simétrico. O limite inferior é truncado em zero porque o
peso vivo não pode ser negativo. A cobertura é calibrada e avaliada por animal,
usando o pior erro quando o mesmo indivíduo possui várias imagens ou visitas.

## Gates obrigatórios

1. qualidade técnica da imagem aprovada pela política presa ao pacote;
2. critérios finais definidos antes da abertura única do teste;
3. IC95% de erro e cobertura dentro dos limites operacionais;
4. validação em outra fazenda, lote ou período;
5. revisão jurídica dos dados e artefatos;
6. revisão de segurança operacional;
7. aprovação humana documentada antes de qualquer liberação.

## Limitações que permanecem

O gate atual mede resolução, proporção, exposição e nitidez. Ele não confirma a
presença de um único bovino, raça, pose lateral, corpo inteiro, oclusão,
distância, câmera ou pertencimento ao domínio validado. Essas verificações
exigem controles adicionais e dados reais.

Mesmo quando o teste técnico passa, o pacote interno não concede autorização
comercial. Uma futura liberação precisa usar outro status, inventário assinado
e revisão separada; este template não é evidência de aprovação.
