"""Relatório das 9 variantes em 15 colunas e no texto de `salvar_metricas`.

Colunas: P@1 P@5 P@10 R@1 R@5 R@10 F1@1 F1@5 F1@10 MRR MAP MRP Setup (ms) Consulta (ms) Total (ms).
"""
KS = (1, 5, 10)
COLUNAS = ([f'P@{k}' for k in KS] + [f'R@{k}' for k in KS] + [f'F1@{k}' for k in KS] +
           ['MRR', 'MAP', 'MRP', 'Setup (ms)', 'Consulta (ms)', 'Total (ms)'])
CHAVE_TEMPO = {'Setup (ms)': 'Setup_ms', 'Consulta (ms)': 'Consulta_ms', 'Total (ms)': 'Total_ms'}


def linha(m):
    """As 15 colunas, nesta ordem, a partir das métricas de uma execução de teste."""
    faltam = [c for c in COLUNAS if CHAVE_TEMPO.get(c, c) not in m]
    if faltam:
        raise KeyError(f'métricas incompletas: faltam {faltam}')
    return {c: m[CHAVE_TEMPO.get(c, c)] for c in COLUNAS}


def texto_metricas(nome, m, n_rep=5):
    """Layout do arquivo de métricas (cabeçalho, bloco de tempo, P/R/F1 por k, MRR, MAP, MRP)."""
    linhas = [f'{nome} — Avaliação Cross-Company', '=' * 55, f"Consultas avaliadas  : {m['n_consultas']}", '',
              f'Tempo de execução (Setup: média de {n_rep} repetições; Consulta: média das consultas do conjunto):',
              f"  {'SETUP (carregamento/preparação, uma vez)':<44}: {m['Setup_ms']:>10.3f} ms",
              f"  {'CONSULTA (por consulta)':<44}: {m['Consulta_ms']:>10.3f} ms",
              f"  {'TOTAL (SETUP + CONSULTA)':<44}: {m['Total_ms']:>10.3f} ms", '']
    for k in KS:
        linhas.append(f"  P@{k:<3}: {m[f'P@{k}']:.4f}  R@{k:<3}: {m[f'R@{k}']:.4f}  F1@{k:<3}: {m[f'F1@{k}']:.4f}")
    linhas += ['', f"  MRR : {m['MRR']:.4f}", f"  MAP : {m['MAP']:.4f}", f"  MRP : {m['MRP']:.4f}", '']
    return '\n'.join(linhas)


def tabela_markdown(linhas_por_variante):
    cab = '| Variante | ' + ' | '.join(COLUNAS) + ' |'
    sep = '|---|' + '---|' * len(COLUNAS)
    corpo = []
    for nome, l in linhas_por_variante.items():
        cel = [f'{l[c]:.4f}' if not c.endswith('(ms)') else f'{l[c]:.1f}' for c in COLUNAS]
        corpo.append(f'| {nome} | ' + ' | '.join(cel) + ' |')
    return '\n'.join([cab, sep] + corpo) + '\n'
