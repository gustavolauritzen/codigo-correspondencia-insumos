"""Cross-Encoder, parte sem GPU. Testável localmente.

monoBERT (Lin et al. 2021, §3.2): a consulta e o candidato entram juntos no BERT e a saída é um score de relevância.
Modelo: BERTimbau Large (config.MODELO_CROSS_ENCODER, mesma família do Bi-Encoder). Treino com os pares de
`dados/treino/` (positivos, negativos do mesmo tipo e negativos de outro tipo); escolha de época e de taxa de
aprendizado na VALIDAÇÃO; teste só na etapa final.
"""
import csv
import os
import random
from collections import defaultdict

import numpy as np

import biencoder_nucleo as BN
import config as C

MAX_LEN_PROVISORIO = 128      # par inteiro (consulta + candidato); confirmado pelo percentil 99 medido com treino e validação
LRS = BN.LRS
EPOCAS = BN.EPOCAS
BATCH = 16
WARMUP = 0.1
SEMENTE = BN.SEMENTE
N_CONSULTAS_VALIDACAO = 100   # amostra fixa de consultas de validação ranqueadas contra o pool inteiro a cada época (custo)
BATCH_PONTUACAO = 256


def ler_pares_ce(formato):
    """Lista de (texto_consulta, texto_candidato, rótulo): positivos (1), negativos do mesmo tipo (0) e negativos de
    outro tipo (0). Só pares de empresas de treino, sem itens vazados (garantido por pares_treino.py)."""
    textos = BN.textos_por_id(formato)
    pasta = os.path.join(C.DB, 'treino')
    saida = []
    for nome, rotulo in (('pares_positivos.csv', 1.0), ('pares_negativos.csv', 0.0), ('pares_negativos_outro_tipo.csv', 0.0)):
        with open(os.path.join(pasta, nome), encoding='utf-8', newline='') as f:
            saida += [(textos[r['id_a']], textos[r['id_b']], rotulo) for r in csv.DictReader(f)]
    return saida


def lotes_ce(pares, tamanho, semente):
    rng = random.Random(semente)
    todos = list(pares)
    rng.shuffle(todos)
    return [todos[i:i + tamanho] for i in range(0, len(todos), tamanho)]


def amostra_validacao(consultas, relevantes, n=N_CONSULTAS_VALIDACAO, semente=SEMENTE):
    """Amostra fixa, repartida igualmente entre as empresas de validação, só consultas com relevante."""
    por_emp = defaultdict(list)
    for q in sorted(consultas, key=lambda c: c['id']):
        if relevantes.get(q['id']):
            por_emp[q['empresa']].append(q)
    rng = random.Random(semente)
    empresas = sorted(por_emp)
    cota = n // len(empresas)
    escolhidas = []
    for e in empresas:
        escolhidas += rng.sample(por_emp[e], min(cota, len(por_emp[e])))
    return sorted(escolhidas, key=lambda c: c['id'])


def montar_par(q_ids, d_ids, cls_id, sep_id, max_len):
    """[CLS] consulta [SEP] candidato [SEP], truncando sempre o texto mais longo até caber em max_len."""
    q, d = list(q_ids), list(d_ids)
    while len(q) + len(d) + 3 > max_len:
        if len(q) >= len(d):
            q.pop()
        else:
            d.pop()
    ids = [cls_id] + q + [sep_id] + d + [sep_id]
    tipos = [0] * (len(q) + 2) + [1] * (len(d) + 1)
    return ids, tipos


def lotes_por_comprimento(comprimentos, tamanho):
    """Índices em lotes de comprimento parecido (menos preenchimento) para pontuar; `restaurar` devolve a ordem original."""
    ordem = np.argsort(np.asarray(comprimentos), kind='stable')
    return [ordem[i:i + tamanho] for i in range(0, len(ordem), tamanho)]


def ranquear_por_scores(ids, scores):
    """Ranking por score decrescente; empate pela ordem de entrada (determinístico)."""
    s = np.asarray(scores, dtype=float)
    ordem = np.lexsort((np.arange(len(s)), -s))
    return [ids[i] for i in ordem]


def percentil(valores, p=99):
    return float(np.percentile(np.asarray(valores), p))


def configuracoes():
    return [{'lr': lr, 'batch': BATCH, 'epocas': EPOCAS, 'warmup': WARMUP, 'semente': SEMENTE,
             'n_consultas_validacao': N_CONSULTAS_VALIDACAO} for lr in LRS]
