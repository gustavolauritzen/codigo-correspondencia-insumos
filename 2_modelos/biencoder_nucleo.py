"""Bi-Encoder, parte sem GPU. Testável localmente.

Tudo o que não precisa de torch: leitura dos pares de treino, lotes sem texto repetido (MNR), ranking e métricas
a partir de vetores, escolha do melhor checkpoint e a lista de configurações da validação.
A parte que usa o modelo está em `biencoder_gpu.py` (roda em GPU).
"""
import csv
import os
import random

import numpy as np

import config as C
import dados
import metricas as M

MAX_SEQ_PROVISORIO = 128      # confirmado ou trocado pelo percentil 99 de tokens medido com treino e validação
LRS = (3e-5, 2e-5)            # 5e-5 fica fora: Devlin et al. (2019), pdf p.6, relatam instabilidade do BERT Large em dados pequenos
EPOCAS = 4                    # avaliado a cada época (1 a 4); a melhor época vale (Devlin: 2, 3 ou 4)
BATCH_COSINE = 16             # Reimers e Gurevych (2019), pdf p.4
BATCH_MNR = 16
ESCALA_MNR = 20.0
N_NEG_POR_POS = 3
SEMENTE = 42


def textos_por_id(formato):
    """Mapa id -> texto de todos os itens da base (formato 'A' = texto_a, 'B' = serialização [COL]/[VAL])."""
    campo = {'A': 'texto_a', 'B': 'texto_b'}[formato]
    mapa = {}
    with open(os.path.join(C.DB, 'base.csv'), encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            mapa[r['id']] = r[campo] if campo in r else None
    return mapa


def ler_pares_treino(formato):
    """(positivos, negativos): listas de (texto_a, texto_b) lidas de dados/treino/."""
    textos = textos_por_id(formato)
    pasta = os.path.join(C.DB, 'treino')

    def ler(nome):
        with open(os.path.join(pasta, nome), encoding='utf-8', newline='') as f:
            return [(textos[r['id_a']], textos[r['id_b']]) for r in csv.DictReader(f)]

    return ler('pares_positivos.csv'), ler('pares_negativos.csv')


def lotes_sem_repeticao(pares, tamanho, semente):
    """Lotes para a MNR: embaralha e monta lotes em que nenhum texto aparece duas vezes (nem como âncora, nem como
    positivo), para um duplicado não virar falso negativo do lote (sentence-transformers, NoDuplicatesDataLoader).
    Pares que não cabem no lote corrente esperam o próximo. Descarta o lote final incompleto."""
    rng = random.Random(semente)
    fila = list(pares)
    rng.shuffle(fila)
    lotes = []
    while fila:
        lote, usados, resto = [], set(), []
        for a, b in fila:
            if len(lote) < tamanho and a not in usados and b not in usados and a != b:
                lote.append((a, b))
                usados.update((a, b))
            else:
                resto.append((a, b))
        if len(lote) < tamanho:
            break                       # o que sobrou não completa um lote
        lotes.append(lote)
        fila = resto
    return lotes


def lotes_com_rotulo(positivos, negativos, tamanho, semente):
    """Lotes para a Cosine: (texto, texto, rótulo 1/0), embaralhados; o último lote incompleto é mantido."""
    rng = random.Random(semente)
    todos = [(a, b, 1.0) for a, b in positivos] + [(a, b, 0.0) for a, b in negativos]
    rng.shuffle(todos)
    return [todos[i:i + tamanho] for i in range(0, len(todos), tamanho)]


def ranquear_vetores(emb_consultas, consultas, emb_cands, cands, topo=None):
    """Ranking por cosseno. Vetores já normalizados ou não (normaliza aqui). Só candidatos de outras empresas.
    Desempate pela ordem do pool. Devolve consulta -> lista de ids."""
    q = emb_consultas / np.maximum(np.linalg.norm(emb_consultas, axis=1, keepdims=True), 1e-12)
    d = emb_cands / np.maximum(np.linalg.norm(emb_cands, axis=1, keepdims=True), 1e-12)
    ids = np.array([c['id'] for c in cands])
    emp = np.array([c['empresa'] for c in cands])
    rankings = {}
    for i, c in enumerate(consultas):
        s = d @ q[i]
        permitido = np.nonzero(emp != c['empresa'])[0]
        ordem = permitido[np.lexsort((permitido, -s[permitido]))]
        if topo is not None:
            ordem = ordem[:topo]
        rankings[c['id']] = list(ids[ordem])
    return rankings


def avaliar_vetores(emb_consultas, consultas, emb_cands, cands, relevantes):
    """Métricas do conjunto a partir dos vetores (só as consultas com relevante)."""
    idx = [i for i, c in enumerate(consultas) if c['id'] in relevantes]
    rk = ranquear_vetores(emb_consultas[idx], [consultas[i] for i in idx], emb_cands, cands)
    return M.avaliar(rk, relevantes)


def escolher_melhor(historico):
    """historico: lista de dicts com 'MAP', 'MRR' e 'epoca' (e o que mais houver). Melhor MAP; empate por MRR; empate
    pela época mais cedo. Critério fixado antes de ver o teste."""
    return max(historico, key=lambda h: (round(h['MAP'], 10), round(h['MRR'], 10), -h['epoca']))


def configuracoes(perda):
    """Configurações da validação para uma perda ('cosine' ou 'mnr'): uma por taxa de aprendizado; as épocas
    são avaliadas dentro de cada execução."""
    return [{'perda': perda, 'lr': lr, 'batch': BATCH_COSINE if perda == 'cosine' else BATCH_MNR, 'epocas': EPOCAS,
             'warmup': 0.1, 'semente': SEMENTE} for lr in LRS]
