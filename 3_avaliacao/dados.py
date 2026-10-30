"""Leitura dos conjuntos de avaliação.

Cada conjunto (validacao, teste) traz consultas, candidatos e gabarito, todos só do próprio conjunto.
A coluna `tipo` de consultas.csv serve ao gabarito e NÃO é entregue às técnicas. O teste só é lido com
`liberar_teste=True` (a etapa final); qualquer outra leitura levanta erro.
"""
import csv
import os
from collections import defaultdict

import config as C


class TesteBloqueado(RuntimeError):
    pass


_TEXTO_B = {}


def _texto_b_da_base():
    """Serialização [COL]/[VAL] (formato B) de cada item da base, para as consultas (consultas.csv só traz o texto A)."""
    if not _TEXTO_B:
        with open(os.path.join(C.DB, 'base.csv'), encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                _TEXTO_B[r['id']] = r['texto_b']
    return _TEXTO_B


def carregar(conjunto, liberar_teste=False):
    if conjunto == 'teste' and not liberar_teste:
        raise TesteBloqueado('o conjunto de teste só é lido na etapa final (liberar_teste=True)')
    if conjunto not in ('validacao', 'teste'):
        raise ValueError(conjunto)
    pasta = os.path.join(C.DB, 'gabarito', conjunto)

    def ler(nome):
        with open(os.path.join(pasta, nome), encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    texto_b = _texto_b_da_base()
    consultas = [{'id': r['query_id'], 'empresa': r['banco_origem'], 'texto': r['texto_a'], 'texto_b': texto_b[r['query_id']]}
                 for r in ler('consultas.csv')]
    candidatos = [{'id': r['doc_id'], 'empresa': r['banco_origem'], 'texto': r['texto_a'], 'texto_b': r['texto_b']}
                  for r in ler('candidatos.csv')]
    relevantes = defaultdict(set)
    for r in ler('gabarito.csv'):
        relevantes[r['query_id']].add(r['doc_id'])
    return consultas, candidatos, dict(relevantes)


def candidatos_da_consulta(consulta, candidatos):
    """Candidatos de outras empresas que a consulta (a comparação é entre empresas)."""
    return [d for d in candidatos if d['empresa'] != consulta['empresa']]
