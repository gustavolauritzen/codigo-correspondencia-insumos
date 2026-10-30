"""Híbridas, parte sem GPU. Testável localmente.

Duas etapas (Lin et al. 2021, §3.4): primeiro estágio (BM25, Bi zero-shot, Bi Cosine ou Bi MNR) entrega os K melhores
candidatos e o Cross-Encoder os reordena. O ranking final tem só esses K (o que ficou de fora conta como não recuperado).
O K é escolhido na VALIDAÇÃO (MAP, desempate MRR, desempate pelo menor K) e depois fixado para o teste.
"""
import os

import metricas as M
import retomada as R

K_GRADE = (20, 50, 100, 200)
TIPOS = ('bm25', 'bi_zero_shot', 'bi_cosine', 'bi_mnr')
NOMES = {'bm25': 'Híbrida BM25 → CE', 'bi_zero_shot': 'Híbrida Bi zero-shot → CE',
         'bi_cosine': 'Híbrida Bi Cosine → CE', 'bi_mnr': 'Híbrida Bi MNR → CE'}


def reordenar(ids_primeiro_estagio, scores_ce, k):
    """Os K primeiros do primeiro estágio, reordenados por score do Cross-Encoder (empate: ordem do primeiro estágio)."""
    ids = list(ids_primeiro_estagio[:k])
    sc = list(scores_ce[:k])
    ordem = sorted(range(len(ids)), key=lambda i: (-sc[i], i))
    return [ids[i] for i in ordem]


def metricas_por_k(top_ids, scores, relevantes, ks=K_GRADE):
    """top_ids/scores: consulta -> listas (primeiro estágio em ordem, scores do CE na mesma ordem), com o K máximo.
    Devolve {K: métricas} sem pontuar de novo (o K menor é prefixo do maior)."""
    saida = {}
    for k in ks:
        rankings = {q: reordenar(top_ids[q], scores[q], k) for q in top_ids}
        saida[k] = M.avaliar(rankings, relevantes)
    return saida


def escolher_k(por_k):
    return max(por_k, key=lambda k: (round(por_k[k]['MAP'], 10), round(por_k[k]['MRR'], 10), -k))


def melhor_execucao(pasta, prefixo):
    """Entre as pastas `prefixo*` já concluídas (por exemplo bi_mnr_lr3e-05_fmtA), a de maior MAP na validação
    (desempate MRR). Devolve (caminho do melhor checkpoint, conteúdo de concluido.json)."""
    candidatas = []
    for nome in sorted(os.listdir(pasta)) if os.path.isdir(pasta) else []:
        if nome.startswith(prefixo) and R.concluido(os.path.join(pasta, nome)):
            c = R.ler_json(os.path.join(pasta, nome, 'concluido.json'))
            candidatas.append((round(c['MAP_validacao'], 10), round(c['MRR_validacao'], 10), nome, c))
    if not candidatas:
        raise FileNotFoundError(f'nenhuma execução concluída com prefixo {prefixo} em {pasta}')
    _, _, nome, c = max(candidatas, key=lambda t: (t[0], t[1], t[2]))
    return os.path.join(pasta, nome, 'melhor'), c
