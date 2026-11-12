"""Gráfico de eficácia e custo: MAP (eixo y) e tempo médio por consulta em escala logarítmica (eixo x) das 9 variantes
no teste, com a fronteira de Pareto (Lin, Nogueira e Yates, 2021). Lê `teste_*/teste.json`.

  python grafico_eficacia_custo.py [saida]      grava eficacia_custo.pdf e .png em saida/analises/
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise_teste as A

ROT = {'Híbrida Bi-Encoder zero-shot': 'Híbrida zero-shot', 'Híbrida Bi-Encoder Cosine': 'Híbrida Cosine',
       'Híbrida Bi-Encoder MNR': 'Híbrida MNR'}   # nas híbridas, o rótulo indica o primeiro estágio
DESL = {'BM25': (7, -14), 'Bi-Encoder zero-shot': (8, -4), 'Bi-Encoder Cosine': (8, -4), 'Bi-Encoder MNR': (-7, 6),
        'Cross-Encoder': (0, -16), 'Híbrida BM25': (-7, -15), 'Híbrida Bi-Encoder zero-shot': (8, -4),
        'Híbrida Bi-Encoder Cosine': (7, -13), 'Híbrida Bi-Encoder MNR': (8, -4)}
ALIN = {'Cross-Encoder': 'center', 'Híbrida BM25': 'right', 'Bi-Encoder MNR': 'right'}


def main(saida):
    pts = []
    for n, p in A.VARIANTES:
        m = json.load(open(os.path.join(saida, p, 'teste.json'), encoding='utf-8'))['metricas']
        pts.append((n, m['Consulta_ms'], m['MAP']))
    fronteira = sorted([a for a in pts if not any(b[1] <= a[1] and b[2] >= a[2] and b != a for b in pts)],
                       key=lambda x: x[1])
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'pdf.fonttype': 42, 'axes.linewidth': 0.9})
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for n, x, y in pts:
        naf = (n, x, y) in fronteira
        ax.scatter(x, y, s=42, marker='o' if naf else 's', facecolors='black' if naf else 'white', edgecolors='black',
                   linewidths=1.1, zorder=3)
        ax.annotate(ROT.get(n, n), (x, y), xytext=DESL[n], textcoords='offset points', ha=ALIN.get(n, 'left'), fontsize=10)
    xs = [f[1] for f in fronteira]; ys = [f[2] for f in fronteira]
    ax.step(xs + [30000], ys + [ys[-1]], where='post', color='black', linestyle='--', linewidth=1.2, zorder=2,
            label='Fronteira de Pareto')
    ax.set_xscale('log'); ax.set_xlim(10, 30000); ax.set_ylim(0.3, 0.9)
    ax.set_xlabel('Tempo médio por consulta (ms, escala logarítmica)'); ax.set_ylabel('MAP')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, color='0.55')
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    fig.tight_layout()
    pasta = os.path.join(saida, 'analises'); os.makedirs(pasta, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(pasta, f'eficacia_custo.{ext}'), dpi=200)
    print('fronteira de Pareto:', [(n, round(x, 1), round(y, 4)) for n, x, y in fronteira])


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'saida')))
