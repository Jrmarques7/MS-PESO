# Proveniência e caminho para o modelo comercial

## B2 atual: somente pesquisa

O checkpoint `b2-cowdb-efficientnet-b0` não está autorizado para uso comercial.
Ele combina:

- implementação EfficientNet-B0 da torchvision, cujo código usa BSD-3-Clause;
- pesos iniciais `IMAGENET1K_V1`;
- ajuste supervisionado com imagens e pesos do CowDB.

O contrato do ImageNet limita o banco de imagens a pesquisa não comercial e
educação. A torchvision também alerta que pesos pré-treinados podem ter termos
derivados do dataset. O CowDB é apresentado como acessível à comunidade de
pesquisa, mas não fornece no repositório uma licença comercial explícita.

Por prudência, nenhum checkpoint descendente do B2 será vendido, incorporado a
serviço comercial, usado como professor de destilação nem empregado para gerar
pseudorrótulos do candidato comercial.

Referências:

- https://image-net.org/accessagreement
- https://github.com/pytorch/vision/blob/main/docs/source/models.rst
- https://github.com/pytorch/vision/blob/main/LICENSE
- https://github.com/ruchaya/CowDB

## Trilha comercial independente

O candidato comercial deve cumprir simultaneamente:

1. arquitetura implementada por código com licença compatível;
2. inicialização aleatória, sem carregar o checkpoint B2 ou pesos ImageNet;
3. treinamento somente em imagens próprias ou expressamente licenciadas;
4. autorização vigente para treinamento de modelos e exploração comercial;
5. rastreabilidade de fazenda, animal, evento, câmera, balança e captura;
6. teste externo mantido fora da seleção de hiperparâmetros;
7. inventário de versões, licenças, autorizações e hashes;
8. revisão jurídica antes do lançamento.

Imagens próprias sem pesagem podem ser usadas em pré-treinamento
auto-supervisionado, desde que possuam os mesmos direitos. Depois, o modelo é
ajustado usando o subconjunto ligado a pesagens confiáveis.

## Gate de promoção

Um pacote comercial futuro deve usar outro `model_id`, outro descritor e outro
checkpoint. `commercial_use_allowed` só poderá ser `true` quando não houver
bloqueios de proveniência, os direitos estiverem documentados e os critérios de
desempenho e segurança no domínio Nelore tiverem sido atingidos.

Este documento registra a regra técnica de governança; não substitui parecer
jurídico nem o instrumento de autorização assinado com cada titular dos dados.
