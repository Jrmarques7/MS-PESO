# API de inferência

## Estado e limite atual

O serviço HTTP está pronto para teste de contrato e futura integração, mas o
descritor versionado no repositório ainda é um candidato não aprovado. Por
padrão, a prontidão responde `503 model_not_promoted` e nenhuma inferência é
exposta. Isso evita transformar acidentalmente o modelo de pesquisa em produto.

O modo `MS_PESO_ALLOW_UNAPPROVED_CANDIDATE=true` existe exclusivamente para um
ambiente interno e isolado de teste. Mesmo nesse modo, cada resposta mantém
`commercial_use_allowed: false` e
`authorization_status: blocked_pending_mandatory_reviews`.

## Instalação e execução

```bash
python -m pip install -e ".[service]"
```

Configure as variáveis tomando `.env.service.example` como referência. A chave
de API deve ter pelo menos 32 caracteres. Em seguida:

```bash
ms-peso-service
```

O padrão escuta somente em `127.0.0.1:8080`. Para expor o processo em uma rede,
configure explicitamente `MS_PESO_HOST` e use TLS em um proxy reverso. Não
registre a chave de API, imagens ou corpos multipart nos logs.

## Rotas

- `GET /health/live`: confirma que o processo está vivo; não exige chave.
- `GET /health/ready`: confirma chave, pacote e modelo carregado; retorna 503
  enquanto alguma condição estiver pendente.
- `GET /v1/model`: estado público do pacote; exige `X-API-Key`.
- `POST /v1/predictions`: recebe `multipart/form-data`; exige `X-API-Key`.
- `POST /v1/video-predictions`: recebe vídeo curto; exige `X-API-Key`.

Campos do `POST /v1/predictions`:

- `image`: JPEG, PNG ou WebP, até 10 MiB por padrão;
- `correlation_id`: identificador opcional do consumidor, até 128 caracteres.

Exemplo:

```bash
curl -X POST http://127.0.0.1:8080/v1/predictions \
  -H "X-API-Key: $MS_PESO_API_KEY" \
  -F "correlation_id=farmup-lote-42-animal-7" \
  -F "image=@foto-lateral.jpg"
```

A API é stateless: não conhece fazenda, lote ou animal e não guarda a foto. O
arquivo é criado com nome aleatório, processado e removido ao final da
requisição. O consumidor é responsável por persistir o vínculo de domínio e o
resultado que desejar auditar.

Uma predição concluída retorna peso pontual, intervalo conformal, qualidade da
imagem, identidade e hashes do modelo. Uma imagem reprovada pela política
retorna HTTP 422 com o mesmo contrato e `prediction_status: rejected`. Falhas
de autenticação usam 401; formato inválido, 415; excesso de tamanho, 413; modelo
indisponível ou bloqueado, 503.

A chave e o limite total da requisição são verificados antes do parser
multipart. O limite exato da imagem é conferido novamente durante a cópia para
o arquivo temporário.

## Contrato de integração

O consumidor deve:

1. tratar `prediction_id` como identidade única do resultado, não como chave de
   idempotência;
2. enviar seu próprio `correlation_id` estável para reconciliar tentativas;
3. guardar os hashes do modelo e da calibração junto da estimativa;
4. mostrar o intervalo, os avisos e a qualidade, não apenas o peso pontual;
5. recusar uso oficial quando `production_ready` ou
   `commercial_use_allowed` forem falsos;
6. nunca converter automaticamente uma estimativa bloqueada em peso de balança;
7. aplicar timeout, retentativa limitada e circuit breaker no cliente.

O serviço usa um único worker por processo porque múltiplos workers duplicam o
modelo na memória da GPU. Escala horizontal deve ser feita com processos ou
instâncias separados e memória suficiente.

## Inferência por vídeo — V1

Campos do `POST /v1/video-predictions`:

- `video`: MP4, MOV, WebM ou AVI, até 50 MiB por padrão;
- `correlation_id`: identificador opcional do consumidor, até 128 caracteres.

Exemplo:

```bash
curl -X POST http://127.0.0.1:8080/v1/video-predictions \
  -H "X-API-Key: $MS_PESO_API_KEY" \
  -F "correlation_id=farmup-lote-42-animal-7-video-1" \
  -F "video=@bovino-lateral.mp4"
```

A política versionada em `configs/video_inference.yaml` limita a duração a dez
segundos, amostra até 20 instantes, exige pelo menos três quadros válidos e
aceita no máximo 8.294.400 pixels por quadro (equivalente a 4K UHD). Seleciona
até cinco quadros com diversidade temporal. O ranqueamento usa somente as
medições técnicas existentes: resolução, proporção, exposição e nitidez. O
vídeo e todos os quadros temporários são removidos ao final da requisição.

O peso retornado é a mediana das estimativas selecionadas. A resposta também
traz dispersão total, desvio absoluto mediano, instante, nota técnica, qualidade
e estimativa de cada quadro. Isso fornece auditoria, mas ainda não valida que o
mesmo bovino aparece em todos os quadros nem que sua pose lateral está correta.

O campo agregado `prediction_interval` permanece nulo com
`interval_status: pending_video_calibration`. Os intervalos dos quadros são
identificados como individuais e não devem ser interpretados como intervalo da
mediana. Da mesma forma, `consensus_status: threshold_not_calibrated` informa
que a divergência ainda não pode aprovar ou reprovar o vídeo. O limiar só será
preenchido depois de uma calibração congelada com vídeos e pesos reais
vinculados; até lá, não inventamos um valor operacional.

Erros de vídeo ilegível, duração excessiva, poucos quadros decodificados,
qualidade insuficiente ou baixa diversidade temporal retornam HTTP 422. O
endpoint herda autenticação, limite antecipado do corpo, bloqueio comercial e
indisponibilidade do modelo do endpoint de imagem.
