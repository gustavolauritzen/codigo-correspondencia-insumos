"""Comprimento em tokens dos textos de TREINO e VALIDAÇÃO, para fixar `max_seq_length` (Bi-Encoder) e `MAX_LEN` (Cross-Encoder).

Usa o tokenizador do BERTimbau Large (o mesmo vocabulário nos dois modelos) e NÃO
abre o teste. Mede, para cada formato (A e B): o texto sozinho (com [CLS] e [SEP]) e o par consulta+candidato (com [CLS] e
dois [SEP]). Decisão: menor valor da grade (128, 192, 256, 512) que cobre o percentil 99.

  python medir_tokens.py <pasta persistente>      # grava tokens_p99.json e registra em decisoes.json
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import json
import os
import random
import sys

import numpy as np

import biencoder_nucleo as BN
import config as C
import dados
import decisoes as D
import retomada as R

SEMENTE = 42
N_PARES_VALIDACAO = 20000


def estatisticas(comprimentos, corte=128):
    a = np.asarray(comprimentos)
    return {'n': int(a.size), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99)),
            'max': int(a.max()), 'fracao_acima_do_corte': float((a > corte).mean()), 'corte': corte}


def textos_treino_e_validacao(formato):
    """(textos soltos, pares (consulta, candidato)) do treino e da validação. Nunca lê o teste."""
    pares_treino = BN.ler_pares_treino(formato)
    pos, neg = pares_treino
    pares = list(pos) + list(neg)
    consultas, cands, rel = dados.carregar('validacao')
    campo = {'A': 'texto', 'B': 'texto_b'}[formato]
    rng = random.Random(SEMENTE)
    q_ok = [q for q in consultas if rel.get(q['id'])]
    for _ in range(N_PARES_VALIDACAO):
        q = rng.choice(q_ok)
        d = rng.choice(cands)
        if d['empresa'] != q['empresa']:
            pares.append((q[campo], d[campo]))
    soltos = {t for p in pares for t in p} | {q[campo] for q in consultas} | {d[campo] for d in cands}
    return sorted(soltos), pares


def medir(persistente):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODELO_CROSS_ENCODER, revision=C.MODELO_CROSS_ENCODER_REV)
    saida = {}
    for formato in ('A', 'B'):
        soltos, pares = textos_treino_e_validacao(formato)
        ids = {t: len(i) for t, i in zip(soltos, tok(soltos, add_special_tokens=False)['input_ids'])}
        solo = [ids[t] + 2 for t in soltos]
        par = [ids[a] + ids[b] + 3 for a, b in pares]
        e_solo, e_par = estatisticas(solo), estatisticas(par)
        saida[formato] = {'texto_solo': e_solo, 'par': e_par,
                          'max_seq': D.comprimento_para(e_solo['p99']), 'max_len': D.comprimento_para(e_par['p99'])}
        D.registrar(persistente, f'max_seq_{formato}', saida[formato]['max_seq'])
        D.registrar(persistente, f'max_len_{formato}', saida[formato]['max_len'])
        print(f"formato {formato}: texto solo p99 = {e_solo['p99']:.0f} (máx {e_solo['max']}) -> max_seq {saida[formato]['max_seq']}; "
              f"par p99 = {e_par['p99']:.0f} (máx {e_par['max']}) -> max_len {saida[formato]['max_len']}")
    R.gravar_json(os.path.join(persistente, 'tokens_p99.json'), saida)
    return saida


if __name__ == '__main__':
    medir(sys.argv[1])
