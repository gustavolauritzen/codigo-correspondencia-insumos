# dados/gabarito/

O gabarito não vem no repositório. Ele entra aqui, depois do pré-processamento (`dados/base.csv`) e antes de qualquer modelo. Como construí-lo fica por sua conta: o trabalho usou um anotador por regras, conferido manualmente em uma amostra, mas qualquer procedimento serve.

Regras que o código assume:

- A divisão em treino, validação e teste é por empresa inteira. Nenhuma empresa de validação ou de teste pode contribuir com itens ao treino.
- Cada empresa de um conjunto é consultante contra as outras empresas do mesmo conjunto. O pool nunca contém itens de outro conjunto.
- Itens do treino com texto igual ao de um item de validação ou de teste devem ser removidos do treino.
- Todo `id` precisa existir em `dados/base.csv`.

Dois conjuntos, `validacao/` e `teste/`, cada um com três arquivos CSV:

| Arquivo | Colunas lidas |
|---|---|
| `consultas.csv` | `query_id`, `banco_origem`, `texto_a` (copie de `base.csv`) |
| `candidatos.csv` | `doc_id`, `banco_origem`, `texto_a`, `texto_b` (copie de `base.csv`) |
| `gabarito.csv` | `query_id`, `doc_id`: um par por linha, só os relevantes |

Outras colunas são ignoradas. O conjunto de teste só é lido na etapa final (`executar.py teste --confirmo-teste`).
