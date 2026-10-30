"""BM25: biblioteca `rank-bm25` (`BM25Okapi`, versão 0.2.2), sem alterar nada do que ela implementa.

A biblioteca calcula, para cada documento d e consulta q (Lin et al., 2021, equação 2):

    BM25(q, d) = soma_{t em q} log((N - df_t + 0,5) / (df_t + 0,5)) * tf_td (k1 + 1) / (tf_td + k1 (1 - b + b L_d / L_ave))

com duas particularidades da própria biblioteca, mantidas: o IDF negativo (termo em mais da metade dos documentos) é trocado
por `epsilon` vezes o IDF médio (`epsilon` = 0,25, o padrão) e cada ocorrência de um termo repetido na consulta soma de novo.

O que é nosso, em volta da biblioteca: o tokenizador (minúsculas, sem acento, letras e dígitos, filtro de palavras funcionais
normalizado nos dois lados), o índice construído sobre o conjunto de candidatos do conjunto (N, df e L_ave do próprio conjunto,
sem rótulos), a exclusão dos candidatos da empresa da consulta, score 0 = não recuperado e o desempate pela ordem do conjunto
(determinístico). A versão anterior, implementação própria da equação (11.32) de Manning, Raghavan e Schütze (2008), está em
a implementação própria do BM25 (equação 11.32 de Manning) foi descartada em favor da biblioteca.
"""
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

import texto as T

_AQUI = os.path.dirname(os.path.abspath(__file__))


def carregar_stopwords(caminho=None):
    caminho = caminho or os.path.join(_AQUI, 'stopwords_pt.txt')
    with open(caminho, encoding='utf-8') as f:
        return frozenset(l.strip() for l in f if l.strip() and not l.startswith('#'))


STOPWORDS = carregar_stopwords()


def tokenizar(texto, usar_stopwords=True):
    toks = re.findall(r'[a-z0-9]+', T.normalizar(texto))
    return [t for t in toks if not (usar_stopwords and t in STOPWORDS)]


class BM25:
    def __init__(self, ids, empresas, textos, k1=1.5, b=0.75, usar_stopwords=True):
        self.ids = list(ids)
        self.empresas = list(empresas)
        self._empresas_unicas = sorted(set(self.empresas))
        self._cod_empresa = np.array([self._empresas_unicas.index(e) for e in self.empresas])
        self.k1, self.b, self.usar_stopwords = k1, b, usar_stopwords
        docs = [tokenizar(t, usar_stopwords) for t in textos]
        self.n = len(docs)
        self._bm25 = BM25Okapi(docs, k1=k1, b=b)      # epsilon padrão da biblioteca (0,25)

    def ajustar(self, k1, b):
        """Troca k1 e b sem refazer o índice (a biblioteca lê os dois a cada consulta; usado na grade da validação)."""
        self.k1, self.b = k1, b
        self._bm25.k1, self._bm25.b = k1, b

    def pontuar(self, consulta):
        return np.asarray(self._bm25.get_scores(tokenizar(consulta, self.usar_stopwords)), dtype=np.float64)

    def ranquear(self, consulta, empresa_consulta, topo=None):
        s = self.pontuar(consulta)
        if empresa_consulta in self._empresas_unicas:
            s[self._cod_empresa == self._empresas_unicas.index(empresa_consulta)] = 0.0
        pos = np.nonzero(s > 0)[0]
        ordem = pos[np.lexsort((pos, -s[pos]))]          # score decrescente, empate pela ordem do pool
        if topo is not None:
            ordem = ordem[:topo]
        return [self.ids[i] for i in ordem], [float(s[i]) for i in ordem]
