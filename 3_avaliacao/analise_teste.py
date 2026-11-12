"""Leitura comum das análises do teste: as consultas do gabarito de teste e o resultado de cada consulta das 9 variantes
(`saida/teste_<variante>/consultas.csv`, gravado pela avaliação do teste). Só lê o que já foi gravado.

O `consultas.csv` do gabarito de teste precisa da coluna `tipo` (tipo de insumo de cada consulta) para as análises por
tipo, além de `query_id` e `banco_origem` (ver dados/gabarito/LEIA-ME.md).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import config as C

VARIANTES = [
    ('BM25', 'teste_bm25'),
    ('Bi-Encoder zero-shot', 'teste_bi_zero_shot'),
    ('Bi-Encoder Cosine', 'teste_bi_cosine'),
    ('Bi-Encoder MNR', 'teste_bi_mnr'),
    ('Cross-Encoder', 'teste_ce_completo'),
    ('Híbrida BM25', 'teste_hibrida_bm25'),
    ('Híbrida Bi-Encoder zero-shot', 'teste_hibrida_bi_zero_shot'),
    ('Híbrida Bi-Encoder Cosine', 'teste_hibrida_bi_cosine'),
    ('Híbrida Bi-Encoder MNR', 'teste_hibrida_bi_mnr'),
]


def ler_consultas():
    with open(os.path.join(C.DB, 'gabarito', 'teste', 'consultas.csv'), encoding='utf-8') as f:
        return {r['query_id']: r for r in csv.DictReader(f)}


def ler_variante(saida, pasta):
    with open(os.path.join(saida, pasta, 'consultas.csv'), encoding='utf-8') as f:
        return {r['query_id']: {k: float(v) for k, v in r.items() if k not in ('query_id', 'gpu')}
                for r in csv.DictReader(f)}


def carregar(saida):
    """(consultas, resultados por variante, ids ordenados). Exige que as 9 variantes tenham as mesmas consultas."""
    cons = ler_consultas()
    dados = {n: ler_variante(saida, p) for n, p in VARIANTES}
    ids = sorted(q for q in cons if q in dados[VARIANTES[0][0]])
    for n, d in dados.items():
        assert sorted(d) == ids, n
    return cons, dados, ids
