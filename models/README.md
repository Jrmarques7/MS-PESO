# Pacotes de modelo

Os arquivos YAML deste diretório versionam identidade, domínio, entrada,
limitações e hash do checkpoint. Os pesos permanecem fora do Git em
`artifacts/` ou em armazenamento de modelos.

Um pacote só pode ser carregado quando:

- o checkpoint existe;
- o SHA-256 coincide;
- arquitetura e tamanho de entrada coincidem com o checkpoint;
- o model card referenciado existe.

O pacote `b2_cowdb.yaml` é experimental e não autoriza uso em produção.
