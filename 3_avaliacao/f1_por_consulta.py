"""F1-Score@k como média das F1 de cada consulta (Opitz e Burst, 2019), a forma reportada no trabalho.

Por consulta q e corte k: P@k = h_k / k, R@k = h_k / |Rel_q| (h_k = relevantes entre os k primeiros, gravado em
`consultas.csv` de cada variante) e F1@k_q = 2 P R / (P + R), com F1 = 0 quando h_k = 0. A F1@k é a média das F1@k_q.
`calcular(d, ids)` devolve {'F1@k': média das F1 por consulta} a partir das linhas de uma variante.

  python f1_por_consulta.py [saida]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_teste as A

KS = (1, 5, 10)


def calcular(d, ids, ks=KS):
    out = {}
    for k in ks:
        h = np.array([d[q][f'h{k}'] for q in ids])
        r = np.array([d[q]['n_rel'] for q in ids])
        p = h / k; rc = h / r
        f1q = np.where(h > 0, 2 * p * rc / np.where(p + rc > 0, p + rc, 1),
                       0.0)
        media_f1 = f1q.mean()
        out[f'F1@{k}'] = float(media_f1)
    return out


def main(saida):
    _, dados, ids = A.carregar(saida)
    for n, _ in A.VARIANTES:
        f = calcular(dados[n], ids)
        print(f'{n:30} ' + ' '.join(f'{c} {v:.4f}' for c, v in f.items()))


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'saida')))
