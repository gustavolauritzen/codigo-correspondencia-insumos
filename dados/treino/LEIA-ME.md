# dados/treino/

Pares de treino, também por sua conta. Cada arquivo tem as colunas `id_a`, `id_b` (ids de `dados/base.csv`, de empresas de treino diferentes):

| Arquivo | Conteúdo | Rótulo no treino |
|---|---|---|
| `pares_positivos.csv` | itens equivalentes | 1 |
| `pares_negativos.csv` | itens do mesmo tipo, não equivalentes | 0 |
| `pares_negativos_outro_tipo.csv` | itens de tipos diferentes (só o Cross-Encoder usa) | 0 |

No trabalho, os negativos eram 3 por positivo. Uma coluna `tipo` pode existir, mas é ignorada.
