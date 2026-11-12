"""Testes pareados e intervalos de confiança da MAP no teste (Morettin e Bussab, 2010).

- Teste dos postos sinalizados de Wilcoxon sobre a AP de cada consulta, nos 36 pares de variantes, com a correção de
  Bonferroni (p vezes o número de comparações).
- O mesmo teste sobre a média da AP de cada tipo de insumo (cada tipo vale como uma observação), porque consultas do
  mesmo tipo não são independentes.
- IC de 95% da MAP pela aproximação normal com o erro padrão estimado por bootstrap (10.000 reamostras, semente 42),
  reamostrando as consultas e, à parte, os tipos inteiros.

  python significancia.py [saida]      grava ic_ap.csv e pares_ap.csv em saida/analises/
"""
import csv
import itertools
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_teste as A

B, SEMENTE = 10000, 42


def main(saida, metrica='ap'):
    cons, dados, ids = A.carregar(saida)
    rng = np.random.default_rng(SEMENTE)
    nomes = [n for n, _ in A.VARIANTES]
    X = {n: np.array([dados[n][q][metrica] for q in ids]) for n in nomes}
    tipos = np.array([cons[q]['tipo'] for q in ids]); ut = sorted(set(tipos))
    idx_tipo = [np.where(tipos == t)[0] for t in ut]
    T = {n: np.array([X[n][ix].mean() for ix in idx_tipo]) for n in nomes}

    # IC da MAP: erro padrão por bootstrap e aproximação normal
    boot_q = rng.integers(0, len(ids), size=(B, len(ids)))
    boot_t = rng.integers(0, len(ut), size=(B, len(ut)))
    ic = []
    for n in nomes:
        mq = X[n][boot_q].mean(axis=1)
        soma = np.array([X[n][ix].sum() for ix in idx_tipo]); cnt = np.array([len(ix) for ix in idx_tipo])
        mt = soma[boot_t].sum(axis=1) / cnt[boot_t].sum(axis=1)
        ep_q, ep_t = mq.std(ddof=1), mt.std(ddof=1); mu = X[n].mean()
        ic.append((n, mu, ep_q, mu - 1.96 * ep_q, mu + 1.96 * ep_q, ep_t, mu - 1.96 * ep_t, mu + 1.96 * ep_t))

    # pares
    linhas = []
    for a, b in itertools.combinations(nomes, 2):
        d = X[a] - X[b]; dt = T[a] - T[b]
        w = stats.wilcoxon(X[a], X[b], zero_method='wilcox').pvalue
        wt = stats.wilcoxon(T[a], T[b], zero_method='wilcox').pvalue
        linhas.append(dict(a=a, b=b, dif=d.mean(), melhor_a=int((d > 0).sum()), melhor_b=int((d < 0).sum()),
                           empate=int((d == 0).sum()), p_wilcoxon=w, tipos_a=int((dt > 0).sum()),
                           tipos_b=int((dt < 0).sum()), p_wilcoxon_tipo=wt))
    for chave in ('p_wilcoxon', 'p_wilcoxon_tipo'):
        for l in linhas:
            l[chave + '_bonferroni'] = min(1.0, l[chave] * len(linhas))

    pasta = os.path.join(saida, 'analises'); os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, f'ic_{metrica}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['variante', 'media', 'ep_boot_consulta', 'ic_consulta_inf', 'ic_consulta_sup',
                                       'ep_boot_tipo', 'ic_tipo_inf', 'ic_tipo_sup'])
        w.writerows([[n] + [f'{v:.4f}' for v in vs] for n, *vs in ic])
    with open(os.path.join(pasta, f'pares_{metrica}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    return ic, linhas


if __name__ == '__main__':
    ic, linhas = main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'saida')))
    for n, mu, _, a, b, _, c, d in ic:
        print(f'{n:30s} MAP {mu:.4f}  IC consulta [{a:.4f}; {b:.4f}]  IC tipo [{c:.4f}; {d:.4f}]')
