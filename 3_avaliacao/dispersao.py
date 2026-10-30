"""Dispersão da MAP entre as consultas no teste (variância e desvio padrão da AP),.

  python dispersao.py <pasta com as pastas teste_*>

Para cada uma das 9 variantes lê `teste_<variante>/consultas.csv` (uma linha por consulta, gravada pela avaliação do teste) e:
  1. recalcula TODAS as métricas médias com `metricas.agregar` e confere contra o `teste.json` da variante
     (P@k, R@k, F1@k, MRR, MAP e MRP), parando se alguma diferir mais que 1e-9;
  2. calcula, sobre a AP por consulta, a variância amostral s2 = soma((AP_i - MAP)^2) / (n - 1) e o desvio
     padrão s = raiz(s2), com n = número de consultas (Bessel; Reichel, 2025; Wazlawick, 2009);
  3. confere s2 e s contra `statistics.variance` e `statistics.stdev` da biblioteca padrão;
  4. grava, na mesma pasta, `dispersao_map_teste.csv` e `dispersao_map_teste.md`.
Não abre o teste nem altera nada já gravado além desses dois arquivos. Só usa a biblioteca padrão.
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import csv
import json
import math
import os
import statistics
import sys

import metricas as M

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
TOLERANCIA = 1e-9


def variancia_amostral(valores):
    """s2 = soma((x_i - media)^2) / (n - 1); exige n >= 2."""
    n = len(valores)
    if n < 2:
        raise ValueError('a variância amostral exige ao menos 2 valores')
    media = sum(valores) / n
    return sum((x - media) ** 2 for x in valores) / (n - 1)


def ler_consultas(arquivo):
    with open(arquivo, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def conferir_metricas(nome, linhas, gravadas):
    """Recalcula as métricas médias a partir das linhas por consulta e compara com as do teste.json."""
    novas = M.agregar(linhas)
    for chave, valor in novas.items():
        if chave == 'n_consultas':
            if valor != gravadas[chave]:
                raise RuntimeError(f'{nome}: n_consultas {valor} != {gravadas[chave]}')
        elif abs(valor - gravadas[chave]) > TOLERANCIA:
            raise RuntimeError(f'{nome}: {chave} recalculada {valor!r} difere da gravada {gravadas[chave]!r}')
    return novas


def calcular(pasta):
    saida = []
    for nome, sub in VARIANTES:
        linhas = ler_consultas(os.path.join(pasta, sub, 'consultas.csv'))
        with open(os.path.join(pasta, sub, 'teste.json'), encoding='utf-8') as f:
            gravadas = json.load(f)['metricas']
        novas = conferir_metricas(nome, linhas, gravadas)
        aps = [float(r['ap']) for r in linhas]
        n = len(aps)
        var = variancia_amostral(aps)
        dp = math.sqrt(var)
        if abs(var - statistics.variance(aps)) > TOLERANCIA or abs(dp - statistics.stdev(aps)) > TOLERANCIA:
            raise RuntimeError(f'{nome}: variância ou desvio padrão diferem da biblioteca statistics')
        saida.append({'Variante': nome, 'n': n, 'MAP': novas['MAP'], 'Variancia': var, 'DesvioPadrao': dp})
    ns = {l['n'] for l in saida}
    if len(ns) > 1:
        raise RuntimeError(f'as variantes avaliaram números diferentes de consultas: {ns}')
    return saida


def br(x, casas=4):
    return f'{x:.{casas}f}'.replace('.', ',')


def gravar(pasta, saida):
    with open(os.path.join(pasta, 'dispersao_map_teste.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['Variante', 'n_consultas', 'MAP', 'Variancia_AP', 'DesvioPadrao_AP'])
        for l in saida:
            w.writerow([l['Variante'], l['n'], f"{l['MAP']:.10f}", f"{l['Variancia']:.10f}", f"{l['DesvioPadrao']:.10f}"])
    linhas = ['| Variante | MAP | Desvio padrão | Variância |', '|---|---|---|---|']
    for l in saida:
        linhas.append(f"| {l['Variante']} | {br(l['MAP'])} | {br(l['DesvioPadrao'])} | {br(l['Variancia'])} |")
    texto = '\n'.join(linhas) + '\n\nVariância amostral (divisor n - 1) da AP entre as consultas do conjunto de teste.\n'
    with open(os.path.join(pasta, 'dispersao_map_teste.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(texto)
    return texto


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(gravar(sys.argv[1], calcular(sys.argv[1])))
