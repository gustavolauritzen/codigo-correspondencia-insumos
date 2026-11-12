"""MAP por empresa consultante e por tipo de insumo, e distribuição da AP (porcentagem de consultas com AP = 1 e com
AP < 0,5) das 9 variantes no teste. Só lê os resultados gravados.

  python por_grupo.py [saida]      grava map_por_grupo.csv, map_por_tipo.csv e distribuicao_ap.csv em saida/analises/
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_teste as A


def media(d, qs, m='ap'):
    return float(np.mean([d[q][m] for q in qs])) if qs else float('nan')


def main(saida):
    cons, dados, ids = A.carregar(saida)
    nomes = [n for n, _ in A.VARIANTES]
    por_emp = defaultdict(list); por_tipo = defaultdict(list)
    for q in ids:
        por_emp[cons[q]['banco_origem']].append(q); por_tipo[cons[q]['tipo']].append(q)
    grupos = {'Todas': ids, **{e: por_emp[e] for e in sorted(por_emp, key=lambda e: -len(por_emp[e]))}}
    pasta = os.path.join(saida, 'analises'); os.makedirs(pasta, exist_ok=True)

    with open(os.path.join(pasta, 'map_por_grupo.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['grupo', 'n'] + nomes)
        for g, qs in grupos.items():
            w.writerow([g, len(qs)] + [f'{media(dados[n], qs):.4f}' for n in nomes])

    venc = defaultdict(int)
    with open(os.path.join(pasta, 'map_por_tipo.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['tipo', 'n_consultas'] + nomes + ['melhor'])
        for t in sorted(por_tipo, key=lambda t: -len(por_tipo[t])):
            ms = [media(dados[n], por_tipo[t]) for n in nomes]
            melhor = nomes[int(np.argmax(ms))]; venc[melhor] += 1
            w.writerow([t, len(por_tipo[t])] + [f'{m:.4f}' for m in ms] + [melhor])
    print(f'Variante de maior MAP em cada um dos {len(por_tipo)} tipos:', dict(venc))

    with open(os.path.join(pasta, 'distribuicao_ap.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['variante', 'MAP', 'pct_AP_1', 'pct_AP_menor_0.5', 'pct_AP_0', 'desvio_padrao'])
        for n in nomes:
            ap = np.array([dados[n][q]['ap'] for q in ids])
            lin = [ap.mean(), 100 * (ap >= 1 - 1e-12).mean(), 100 * (ap < 0.5).mean(), 100 * (ap == 0).mean(),
                   ap.std(ddof=1)]
            w.writerow([n] + [f'{x:.4f}' for x in lin])
            print(f'{n:30} MAP {lin[0]:.4f}  AP=1 {lin[1]:5.1f}%  AP<0,5 {lin[2]:5.1f}%  AP=0 {lin[3]:5.1f}%')


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'saida')))
