# dados/bruto/

Coleta bruta, fornecida por você (não vem no repositório).

- `ranking_empresas_2026.csv`: colunas `nome_ficticio` e `itens_ordem`. Empresas com `itens_ordem` acima de `CORTE_ITENS_ORDEM` (config.py) entram.
- `base_dados_insumos.csv`: uma linha por insumo, com as colunas `id`, `banco_origem` (nome da empresa, como em `nome_ficticio`), `nome_item`, `descricao_item`, `unidade` e as colunas que o pipeline remove (`nome_insumo`, `palavras_chave`, `tamanho`, `referencia_item`).

`python executar.py preprocessar` lê esses arquivos e grava `dados/base.csv`.
