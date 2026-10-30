"""Testes da parte sem GPU do Cross-Encoder."""
import os
import sys
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TEM_DADOS = all(os.path.exists(os.path.join(_RAIZ, 'dados', *p)) for p in (
    ('base.csv',), ('gabarito', 'validacao', 'gabarito.csv'), ('treino', 'pares_positivos.csv')))
SEM_DADOS = 'precisa de dados/ (base, gabarito e pares de treino, feitos por quem executa)'
import crossencoder_nucleo as X
import dados


class TestMontagem(unittest.TestCase):
    def test_formato_bert(self):
        ids, tipos = X.montar_par([10, 11], [20, 21, 22], 1, 2, 128)
        self.assertEqual(ids, [1, 10, 11, 2, 20, 21, 22, 2])
        self.assertEqual(tipos, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_truncamento_corta_o_mais_longo_e_respeita_o_limite(self):
        ids, tipos = X.montar_par(list(range(10, 20)), list(range(100, 104)), 1, 2, 10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(ids), len(tipos))
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(ids.count(2), 2)
        sep = ids.index(2)
        self.assertTrue(0 < sep < len(ids) - 1)           # os dois textos continuam presentes

    def test_lotes_por_comprimento_cobrem_tudo(self):
        lotes = X.lotes_por_comprimento([5, 1, 9, 3, 7], 2)
        self.assertEqual(sorted(i for l in lotes for i in l), [0, 1, 2, 3, 4])
        self.assertEqual(list(lotes[0]), [1, 3])

    def test_ranking_por_scores_com_empate(self):
        self.assertEqual(X.ranquear_por_scores(['a', 'b', 'c', 'd'], [0.1, 0.9, 0.9, 0.5]), ['b', 'c', 'd', 'a'])


class TestAmostraEPares(unittest.TestCase):
    def test_amostra_fixa_e_equilibrada(self):
        cons = [{'id': f'{e}{i}', 'empresa': e} for e in 'ABC' for i in range(50)]
        rel = {c['id']: {'x'} for c in cons}
        a1 = X.amostra_validacao(cons, rel, 30, 1)
        a2 = X.amostra_validacao(cons, rel, 30, 1)
        self.assertEqual(a1, a2)
        self.assertEqual(len(a1), 30)
        self.assertEqual({e: sum(1 for c in a1 if c['empresa'] == e) for e in 'ABC'}, {'A': 10, 'B': 10, 'C': 10})

    def test_amostra_so_com_relevante(self):
        cons = [{'id': 'q1', 'empresa': 'A'}, {'id': 'q2', 'empresa': 'A'}]
        self.assertEqual([c['id'] for c in X.amostra_validacao(cons, {'q1': {'x'}}, 10, 1)], ['q1'])

    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_amostra_de_validacao_real(self):
        cons, cands, rel = dados.carregar('validacao')
        a = X.amostra_validacao(cons, rel)
        self.assertEqual(len(a), 99)                                   # 100 // 3 por empresa x 3
        self.assertTrue(all(rel[c['id']] for c in a))

    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_pares_de_treino_do_ce(self):
        pares = X.ler_pares_ce('A')
        pos = sum(1 for *_, r in pares if r == 1.0)
        self.assertEqual(pos, 13476)
        self.assertGreater(len(pares) - pos, 40000)
        self.assertTrue(all(a and b for a, b, _ in pares[:300]))

    def test_configuracoes(self):
        self.assertEqual([c['lr'] for c in X.configuracoes()], [3e-5, 2e-5])


if __name__ == '__main__':
    unittest.main()
