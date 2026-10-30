"""Testes da parte sem GPU do Bi-Encoder."""
import os
import sys
import unittest

import numpy as np

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TEM_DADOS = all(os.path.exists(os.path.join(_RAIZ, 'dados', *p)) for p in (
    ('base.csv',), ('gabarito', 'validacao', 'gabarito.csv'), ('treino', 'pares_positivos.csv')))
SEM_DADOS = 'precisa de dados/ (base, gabarito e pares de treino, feitos por quem executa)'
import biencoder_nucleo as N


class TestLotes(unittest.TestCase):
    def test_mnr_sem_texto_repetido_no_lote(self):
        pares = [(f'a{i % 5}', f'b{i}') for i in range(40)]          # âncoras repetidas
        lotes = N.lotes_sem_repeticao(pares, 4, semente=1)
        self.assertTrue(lotes)
        for lote in lotes:
            textos = [t for p in lote for t in p]
            self.assertEqual(len(textos), len(set(textos)))
            self.assertEqual(len(lote), 4)

    def test_mnr_reprodutivel_e_sem_perda_de_par_alem_do_resto(self):
        pares = [(f'a{i}', f'b{i}') for i in range(32)]
        l1 = N.lotes_sem_repeticao(pares, 8, semente=3)
        l2 = N.lotes_sem_repeticao(pares, 8, semente=3)
        self.assertEqual(l1, l2)
        self.assertEqual(sum(len(l) for l in l1), 32)

    def test_mnr_descarta_par_com_textos_iguais(self):
        lotes = N.lotes_sem_repeticao([('x', 'x'), ('a', 'b'), ('c', 'd')], 2, semente=0)
        self.assertEqual([p for l in lotes for p in l if p == ('x', 'x')], [])

    def test_cosine_rotulos(self):
        lotes = N.lotes_com_rotulo([('p1', 'p2')] * 3, [('n1', 'n2')] * 5, 4, semente=0)
        todos = [x for l in lotes for x in l]
        self.assertEqual(sum(1 for *_, r in todos if r == 1.0), 3)
        self.assertEqual(sum(1 for *_, r in todos if r == 0.0), 5)
        self.assertEqual(len(lotes), 2)


class TestRanking(unittest.TestCase):
    def test_cosseno_e_exclusao_da_empresa(self):
        cands = [{'id': 'd1', 'empresa': 'B'}, {'id': 'd2', 'empresa': 'B'}, {'id': 'd3', 'empresa': 'A'}]
        emb_d = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]])
        cons = [{'id': 'q', 'empresa': 'A'}]
        emb_q = np.array([[2.0, 0.0]])                      # a norma não importa
        rk = N.ranquear_vetores(emb_q, cons, emb_d, cands)
        self.assertEqual(rk['q'], ['d1', 'd2'])             # d3 é da empresa da consulta

    def test_desempate_pela_ordem_do_pool(self):
        cands = [{'id': 'd1', 'empresa': 'B'}, {'id': 'd2', 'empresa': 'B'}]
        rk = N.ranquear_vetores(np.array([[1.0, 0.0]]), [{'id': 'q', 'empresa': 'A'}], np.array([[1.0, 0.0], [1.0, 0.0]]), cands)
        self.assertEqual(rk['q'], ['d1', 'd2'])

    def test_metricas_a_partir_dos_vetores(self):
        cands = [{'id': 'd1', 'empresa': 'B'}, {'id': 'd2', 'empresa': 'B'}]
        cons = [{'id': 'q1', 'empresa': 'A'}, {'id': 'q2', 'empresa': 'A'}]
        out = N.avaliar_vetores(np.array([[1.0, 0.0], [1.0, 0.0]]), cons, np.array([[1.0, 0.0], [0.0, 1.0]]), cands,
                                {'q1': {'d1'}})                     # q2 não tem relevante: fora
        self.assertEqual(out['n_consultas'], 1)
        self.assertEqual(out['MRR'], 1.0)


class TestEscolha(unittest.TestCase):
    def test_map_depois_mrr_depois_epoca_cedo(self):
        h = [{'epoca': 1, 'MAP': 0.5, 'MRR': 0.7}, {'epoca': 2, 'MAP': 0.6, 'MRR': 0.7},
             {'epoca': 3, 'MAP': 0.6, 'MRR': 0.8}, {'epoca': 4, 'MAP': 0.6, 'MRR': 0.8}]
        self.assertEqual(N.escolher_melhor(h)['epoca'], 3)

    def test_configuracoes(self):
        cfg = N.configuracoes('mnr')
        self.assertEqual([c['lr'] for c in cfg], [3e-5, 2e-5])
        self.assertTrue(all(c['epocas'] == 4 and c['warmup'] == 0.1 for c in cfg))


class TestDadosReais(unittest.TestCase):
    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_pares_de_treino_viram_textos(self):
        pos, neg = N.ler_pares_treino('A')
        self.assertEqual(len(pos), 13476)
        self.assertTrue(all(isinstance(a, str) and a and isinstance(b, str) and b for a, b in pos[:200] + neg[:200]))
        posb, _ = N.ler_pares_treino('B')
        self.assertTrue(posb[0][0].startswith('[COL]'))


if __name__ == '__main__':
    unittest.main()
