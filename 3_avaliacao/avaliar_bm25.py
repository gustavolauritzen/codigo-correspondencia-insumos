"""BM25: grade na validação e execução final.

  python avaliar_bm25.py validacao      # grade de k1, b e filtro de palavras funcionais; escolhe por MAP (desempate MRR)
  python avaliar_bm25.py teste          # execução final com a escolha gravada; só depois de todas as técnicas

A escolha fica em `dados/resultados/bm25_escolha.json`; o teste lê esse arquivo e não aceita parâmetros.
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import csv
import itertools
import json
import os
import sys

import bm25
import config as C
import dados
import metricas as M

K1S = (0.9, 1.2, 1.5, 1.8, 2.0)
BS = (0.5, 0.75, 0.9)
STOP = (True, False)


def pasta_resultados():
    p = os.path.join(C.DB, 'resultados')
    os.makedirs(p, exist_ok=True)
    return p


def executar(conjunto, k1, b, usar_stopwords, indice=None):
    """Métricas de uma combinação na VALIDAÇÃO (o teste só é lido por final_teste, com o protocolo de execucao_teste.py)."""
    if conjunto != 'validacao':
        raise ValueError('executar serve só à validação')
    consultas, cands, rel = dados.carregar(conjunto)
    if indice is None:
        indice = bm25.BM25([d['id'] for d in cands], [d['empresa'] for d in cands], [d['texto'] for d in cands],
                           k1=k1, b=b, usar_stopwords=usar_stopwords)
    else:
        indice.ajustar(k1, b)
    rankings = {q['id']: indice.ranquear(q['texto'], q['empresa'])[0] for q in consultas if q['id'] in rel}
    return M.avaliar(rankings, rel), indice


def grade_validacao():
    linhas = []
    indices = {}
    for sw in STOP:
        for k1, b in itertools.product(K1S, BS):
            out, indices[sw] = executar('validacao', k1, b, sw, indice=indices.get(sw))
            linhas.append({'k1': k1, 'b': b, 'stopwords': sw, **out})
            print(f"k1={k1} b={b} stop={sw}: MAP {out['MAP']:.4f} MRR {out['MRR']:.4f}", flush=True)
    melhor = max(linhas, key=lambda r: (round(r['MAP'], 10), round(r['MRR'], 10)))
    with open(os.path.join(pasta_resultados(), 'bm25_grade_validacao.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(linhas)
    escolha = {'k1': melhor['k1'], 'b': melhor['b'], 'stopwords': melhor['stopwords'],
               'MAP_validacao': melhor['MAP'], 'MRR_validacao': melhor['MRR'], 'criterio': 'MAP, desempate MRR'}
    with open(os.path.join(pasta_resultados(), 'bm25_escolha.json'), 'w', encoding='utf-8') as f:
        json.dump(escolha, f, ensure_ascii=False, indent=1)
    print('ESCOLHA:', escolha)
    return escolha


def final_teste(local=None, persistente=None):
    """ETAPA FINAL do BM25 com o protocolo único de execucao_teste.py e a escolha gravada na validação. `local`: pasta de saída;
    `persistente`: pasta opcional para uma cópia conferida."""
    import execucao_teste
    with open(os.path.join(pasta_resultados(), 'bm25_escolha.json'), encoding='utf-8') as f:
        e = json.load(f)
    return execucao_teste.avaliar_teste_tecnica(
        'teste_bm25',
        lambda cands: bm25.BM25([d['id'] for d in cands], [d['empresa'] for d in cands], [d['texto'] for d in cands],
                                k1=e['k1'], b=e['b'], usar_stopwords=e['stopwords']),
        lambda idx, q: idx.ranquear(q['texto'], q['empresa'])[0],
        local or pasta_resultados(), persistente, meta={'tecnica': 'bm25', 'implementacao': 'rank-bm25 0.2.2 (BM25Okapi)', **e})


if __name__ == '__main__':
    alvo = sys.argv[1] if len(sys.argv) > 1 else 'validacao'
    if alvo == 'validacao':
        grade_validacao()
    elif alvo == 'teste':
        print(final_teste())
    else:
        raise SystemExit('uso: avaliar_bm25.py validacao|teste')
