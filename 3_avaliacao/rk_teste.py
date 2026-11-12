"""R@K do primeiro estágio no teste, o teto de cada híbrida: a fração dos relevantes de cada consulta que o primeiro
estágio entrega ao Cross-Encoder com K = 1, 5, 10, 20, 50, 100 e 200. Lê `saida/rank_teste_<tipo>/ranking.jsonl`
(gravado por rankings_teste.primeiro_estagio_teste). Também dá a MAP da híbrida para cada K a partir das pontuações do
Cross-Encoder gravadas (`rank_teste_hibrida_<tipo>`).

  python rk_teste.py [saida]      grava rk_primeiro_estagio_teste.csv em saida/analises/
"""
import csv
import json
import os
import sys

import numpy as np

for _d in ('..', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import dados
import hibrida_nucleo as H

KS = (1, 5, 10, 20, 50, 100, 200)
TIPOS = ('bm25', 'bi_zero_shot', 'bi_cosine', 'bi_mnr')


def ler(saida, pasta):
    top, sc = {}, {}
    for l in open(os.path.join(saida, pasta, 'ranking.jsonl'), encoding='utf-8'):
        r = json.loads(l); top[r['query_id']] = r['ids']; sc[r['query_id']] = r['scores']
    return top, sc


def main(saida):
    _, _, rel = dados.carregar('teste', liberar_teste=True)
    linhas = []
    for tipo in TIPOS:
        top, _ = ler(saida, f'rank_teste_{tipo}')
        for k in KS:
            rk = [len(set(top[q][:k]) & rel[q]) / len(rel[q]) for q in top]
            zero = np.mean([not (set(top[q][:k]) & rel[q]) for q in top]) * 100
            linhas.append([tipo, k, f'{np.mean(rk):.4f}', f'{zero:.2f}'])
            print(f'{tipo:13} K={k:3d}  R@K média {np.mean(rk):.4f}  consultas sem relevante {zero:5.1f}%')
        pasta_h = os.path.join(saida, f'rank_teste_hibrida_{tipo}')
        if os.path.exists(os.path.join(pasta_h, 'ranking.jsonl')):
            topk, sc = ler(saida, f'rank_teste_hibrida_{tipo}')
            kmax = len(next(iter(topk.values())))
            por_k = H.metricas_por_k(topk, sc, rel, ks=tuple(k for k in KS if k <= kmax))
            print('   híbrida, MAP por K:', {k: round(v['MAP'], 4) for k, v in por_k.items()})
    pasta = os.path.join(saida, 'analises'); os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, 'rk_primeiro_estagio_teste.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['primeiro_estagio', 'K', 'R@K_media', 'pct_consultas_sem_relevante'])
        w.writerows(linhas)


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'saida')))
