"""Avaliação humana do primeiro colocado por dois anotadores, às cegas.

  python avaliacao_humana.py planilhas [saida]    gera as duas planilhas cegas e a chave em saida/avaliacao_humana/
  python avaliacao_humana.py concordancia [saida] lê as planilhas preenchidas e calcula as concordâncias

Amostra: as 100 consultas da amostra do Cross-Encoder sobre todos os candidatos (rankings_teste.ce_amostra_teste,
semente 42). Para cada consulta, o primeiro colocado do BM25, do Bi-Encoder MNR e do Cross-Encoder; pares repetidos
entre técnicas entram uma vez. As fichas são embaralhadas (semente 42) e não mostram a técnica nem o rótulo do gabarito.
Cada anotador responde S (o mesmo insumo para fins de compra), N ou ?.

Concordância: kappa de Cohen com P(E) pelas marginais combinadas dos dois julgamentos (Manning; Raghavan; Schütze,
2008), entre os anotadores e de cada um (e do consenso) com o gabarito; fichas com '?' ficam fora. Precisão do
primeiro colocado de cada técnica segundo o gabarito, cada anotador e o consenso (os dois marcaram S).
"""
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

for _d in ('..', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import dados

SEMENTE = 42
TECNICAS = [('BM25', 'rank_teste_bm25'), ('Bi-Encoder MNR', 'rank_teste_bi_mnr'),
            ('Cross-Encoder', 'rank_teste_ce_amostra')]
COL = 'sua_resposta (S/N/?)'


def _top1(saida, pasta):
    top = {}
    for l in open(os.path.join(saida, pasta, 'ranking.jsonl'), encoding='utf-8'):
        r = json.loads(l); top[r['query_id']] = r['ids'][0]
    return top


def planilhas(saida):
    meta = json.load(open(os.path.join(saida, 'rank_teste_ce_amostra', 'meta.json'), encoding='utf-8'))
    amostra = list(meta['amostra'])
    consultas, cands, rel = dados.carregar('teste', liberar_teste=True)
    texto = {**{q['id']: (q['empresa'], q['texto']) for q in consultas}, **{d['id']: (d['empresa'], d['texto']) for d in cands}}
    pares = {}
    for nome, pasta in TECNICAS:
        top = _top1(saida, pasta)
        for q in amostra:
            pares.setdefault((q, top[q]), []).append(nome)
    fichas = sorted(pares)
    random.Random(SEMENTE).shuffle(fichas)
    pasta = os.path.join(saida, 'avaliacao_humana'); os.makedirs(pasta, exist_ok=True)
    cab = ['ficha', 'consulta_id', 'consulta_texto', 'candidato_id', 'candidato_empresa', 'candidato_texto', COL, 'observacao']
    for anot in (1, 2):
        with open(os.path.join(pasta, f'anotador{anot}_planilha.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f); w.writerow(cab)
            for n, (q, d) in enumerate(fichas, 1):
                w.writerow([n, q, texto[q][1], d, texto[d][0], texto[d][1], '', ''])
    with open(os.path.join(pasta, 'chave.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['ficha', 'consulta_id', 'candidato_id', 'tecnicas', 'rotulo_gabarito'])
        for n, (q, d) in enumerate(fichas, 1):
            w.writerow([n, q, d, ';'.join(pares[(q, d)]), 'R' if d in rel[q] else 'N'])
    print(f'{len(amostra)} consultas, {len(fichas)} fichas')


def kappa(a, b):
    n = len(a)
    pa = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(((a.count(c) + b.count(c)) / (2 * n)) ** 2 for c in sorted(set(a) | set(b)))
    return pa, pe, (pa - pe) / (1 - pe)


def concordancia(saida):
    pasta = os.path.join(saida, 'avaliacao_humana')
    ler = lambda nome: {r['ficha']: r for r in csv.DictReader(open(os.path.join(pasta, nome), encoding='utf-8-sig'))}
    ch = ler('chave.csv')
    r1 = {k: v[COL].strip().upper() for k, v in ler('anotador1_planilha.csv').items()}
    r2 = {k: v[COL].strip().upper() for k, v in ler('anotador2_planilha.csv').items()}
    gab = {k: 'S' if v['rotulo_gabarito'] == 'R' else 'N' for k, v in ch.items()}
    fichas = sorted(ch, key=int)
    sn = [k for k in fichas if '?' not in (r1[k], r2[k])]
    for nome, x, y in (('anotador 1 x anotador 2', r1, r2), ('anotador 1 x gabarito', r1, gab), ('anotador 2 x gabarito', r2, gab)):
        pa, pe, k = kappa([x[f] for f in sn], [y[f] for f in sn])
        print(f'{nome}: n = {len(sn)}, P(A) = {pa:.4f}, P(E) = {pe:.4f}, kappa = {k:.4f}')
    cons = {k: 'S' if r1[k] == r2[k] == 'S' else ('N' if r1[k] == r2[k] == 'N' else 'D') for k in fichas}
    ok = [k for k in fichas if cons[k] != 'D']
    pa, _, k = kappa([cons[f] for f in ok], [gab[f] for f in ok])
    print(f'consenso x gabarito: n = {len(ok)}, P(A) = {pa:.4f}, kappa = {k:.4f}')
    por_tec = defaultdict(list)
    for k in fichas:
        for t in ch[k]['tecnicas'].split(';'):
            por_tec[t].append(k)
    for t, _ in TECNICAS:
        ks = por_tec[t]
        print(f"{t:16} primeiro colocado equivalente: gabarito {sum(gab[k] == 'S' for k in ks)}, anotador 1 "
              f"{sum(r1[k] == 'S' for k in ks)}, anotador 2 {sum(r2[k] == 'S' for k in ks)}, consenso {sum(cons[k] == 'S' for k in ks)}")


if __name__ == '__main__':
    acao = sys.argv[1] if len(sys.argv) > 1 else ''
    saida = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'saida'))
    {'planilhas': planilhas, 'concordancia': concordancia}.get(acao, lambda s: print(__doc__))(saida)
