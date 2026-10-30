"""Comparativo das 9 variantes no teste. Lê os `teste.json` de cada variante e grava, na mesma pasta:
`comparativo_teste.csv`, `comparativo_teste.md` e um `<variante>_metricas.txt`.

  python comparar.py <pasta com as pastas teste_*>      # exige as 9 variantes concluídas
Não abre o teste: só lê resultados já gravados.
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import csv
import os
import sys

import relatorio as RL
import retomada as R

VARIANTES = [
    ('BM25', 'teste_bm25'),
    ('Bi-Encoder zero-shot', 'teste_bi_zero_shot'),
    ('Bi-Encoder Cosine', 'teste_bi_cosine'),
    ('Bi-Encoder MNR', 'teste_bi_mnr'),
    ('Cross-Encoder (conjunto completo)', 'teste_ce_completo'),
    ('Híbrida BM25 → CE', 'teste_hibrida_bm25'),
    ('Híbrida Bi zero-shot → CE', 'teste_hibrida_bi_zero_shot'),
    ('Híbrida Bi Cosine → CE', 'teste_hibrida_bi_cosine'),
    ('Híbrida Bi MNR → CE', 'teste_hibrida_bi_mnr'),
]


def coletar(pasta, exigir_todas=True):
    linhas, ausentes, brutas = {}, [], {}
    for nome, subpasta in VARIANTES:
        arq = os.path.join(pasta, subpasta, 'teste.json')
        if not os.path.exists(arq) or not R.concluido(os.path.join(pasta, subpasta)):
            ausentes.append(nome)
            continue
        m = R.ler_json(arq)['metricas']
        brutas[nome] = m
        linhas[nome] = RL.linha(m)
    if exigir_todas and ausentes:
        raise RuntimeError(f'variantes sem teste concluído: {ausentes}')
    ns = {m['n_consultas'] for m in brutas.values()}
    if len(ns) > 1:
        raise RuntimeError(f'as variantes avaliaram números diferentes de consultas: {ns}')
    return linhas, brutas, ausentes


def gerar(pasta, exigir_todas=True):
    linhas, brutas, ausentes = coletar(pasta, exigir_todas)
    with open(os.path.join(pasta, 'comparativo_teste.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['Variante'] + RL.COLUNAS)
        for nome, l in linhas.items():
            w.writerow([nome] + [l[c] for c in RL.COLUNAS])
    R.escrita_atomica(os.path.join(pasta, 'comparativo_teste.md'), RL.tabela_markdown(linhas))
    for nome, sub in VARIANTES:
        if nome in brutas:
            R.escrita_atomica(os.path.join(pasta, f'{sub}_metricas.txt'), RL.texto_metricas(nome, brutas[nome]))
    return linhas, ausentes


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    linhas, _ = gerar(sys.argv[1])
    print(open(os.path.join(sys.argv[1], 'comparativo_teste.md'), encoding='utf-8').read())
