# codigo-correspondencia-insumos

Código do TCC "Avaliação de Técnicas de Correspondência Textual para Automatização do Cruzamento entre Descrições de Insumos da Construção Civil" (UNIVALI, 2026). Compara BM25, Bi-Encoder, Cross-Encoder e abordagens híbridas (9 variantes) na correspondência entre descrições de insumos de empresas diferentes.

O repositório traz o método, não os dados nem o gabarito. Com outro corpus e outro gabarito, os números serão outros.

## Estrutura

- `1_preprocessamento/`: limpeza, texto e deduplicação da base.
- `2_modelos/`: BM25, Bi-Encoder, Cross-Encoder e híbridas.
- `3_avaliacao/`: métricas, protocolo de teste, comparativo.
- `dados/`: entradas (`bruto/`, `gabarito/`, `treino/`); cada pasta tem um `LEIA-ME.md` com o formato.
- `executar.py`: roda cada etapa.

## Como reproduzir

Requisitos: Python 3.13, GPU (os tempos só valem em GPU) e `pip install -r requirements.txt`.

1. Coloque a coleta em `dados/bruto/` (formato em `dados/bruto/LEIA-ME.md`).
2. `python executar.py preprocessar`
3. **Monte o gabarito e os pares de treino.** Fica por sua conta: formato e regras em `dados/gabarito/LEIA-ME.md` e `dados/treino/LEIA-ME.md`.
4. `python executar.py bm25`
5. `python executar.py tokens`
6. `python executar.py formato`
7. `python executar.py treinar`
8. `python executar.py hibridas`
9. `python executar.py teste --confirmo-teste`

O resultado (as 9 variantes) fica em `saida/comparativo_teste.md`. As etapas retomam de onde pararam e nunca refazem o que já concluiu.

Os modelos são o BERTimbau Large (`rufimelo/bert-large-portuguese-cased-sts` e `neuralmind/bert-large-portuguese-cased`), em revisões fixadas no `config.py`. Semente 42.

## Testes

`python -m unittest discover -s tests` (os que exigem `rank-bm25` ou dados reais são pulados quando não houver).
