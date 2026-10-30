"""Configuração única do pipeline: caminhos, corte de empresas, modelos e versões fixadas.

A divisão das empresas em treino, validação e teste e o gabarito não estão aqui: são entradas de quem executa
(ver dados/gabarito/LEIA-ME.md e dados/treino/LEIA-ME.md).
"""
import os

RAIZ = os.path.abspath(os.path.dirname(__file__))
COLETA_DIR = os.path.join(RAIZ, 'dados', 'bruto')
DB = os.path.join(RAIZ, 'dados')

# Corte de empresas (critério usado no trabalho; ajuste ao seu corpus).
PERIODO_RANKING = '2026'
CORTE_ITENS_ORDEM = 1000          # empresas com itens_ordem > corte no ranking
MIN_ITENS_CATALOGO = 100          # exclui catálogo pequeno demais

# Colunas da coleta. A categoria e as colunas abaixo NUNCA entram no pipeline.
COLUNAS_REMOVIDAS = ('nome_insumo', 'palavras_chave', 'tamanho', 'referencia_item')
COLUNAS_MANTIDAS = ('id', 'banco_origem', 'nome_item', 'descricao_item', 'unidade')

# Modelos: identificadores EXPLÍCITOS e sempre BERTimbau Large (24 camadas, hidden_size 1024).
MODELO_BI_ENCODER = 'rufimelo/bert-large-portuguese-cased-sts'
MODELO_CROSS_ENCODER = 'neuralmind/bert-large-portuguese-cased'
# Revisões fixadas no Hugging Face (commit). O Bi-Encoder deriva do BERTimbau Large da neuralmind, refinado em STS
# (ASSIN, ASSIN2, STSb pt). Os dois: 24 camadas, 1024 dimensões, 16 cabeças, vocabulário 29.794.
MODELO_BI_ENCODER_REV = 'e5c615d0dd46078764b7f58835b95af95f153e95'
MODELO_CROSS_ENCODER_REV = 'aa302f6ea73b759f7df9cad58bd272127b67ec28'
BERT_LARGE_HIDDEN = 1024
BERT_LARGE_CAMADAS = 24

KS = (1, 5, 10)

# Expansão de siglas: desligada (não houve ganho na validação).
EXPANDIR_SIGLAS = False
