"""Testes de métricas, BM25, tempo e bloqueio do teste. Valores calculados à mão."""
import math
import os
import sys
import time
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TEM_DADOS = all(os.path.exists(os.path.join(_RAIZ, 'dados', *p)) for p in (
    ('base.csv',), ('gabarito', 'validacao', 'gabarito.csv'), ('treino', 'pares_positivos.csv')))
SEM_DADOS = 'precisa de dados/ (base, gabarito e pares de treino, feitos por quem executa)'
try:
    import bm25
except ImportError:          # rank-bm25 não instalado
    bm25 = None
import dados
import metricas as M
import tempo


class TestMetricas(unittest.TestCase):
    def test_uma_consulta(self):
        rk, rel = ['a', 'b', 'c', 'd'], {'b', 'd'}
        self.assertEqual(M.precisao_em_k(rk, rel, 1), 0.0)
        self.assertAlmostEqual(M.precisao_em_k(rk, rel, 5), 2 / 5)       # só 4 no ranking: divide por k
        self.assertEqual(M.revocacao_em_k(rk, rel, 5), 1.0)
        self.assertEqual(M.rr(rk, rel), 0.5)
        self.assertAlmostEqual(M.ap(rk, rel), (1 / 2 + 2 / 4) / 2)
        self.assertEqual(M.r_precisao(rk, rel), 0.5)

    def test_relevante_nao_recuperado_conta_zero(self):
        self.assertAlmostEqual(M.ap(['a'], {'a', 'z'}), 0.5)
        self.assertEqual(M.rr(['a', 'b'], {'z'}), 0.0)

    def test_f1_das_medias(self):
        rks = {'q1': ['a', 'x'], 'q2': ['y', 'z']}
        rel = {'q1': {'a'}, 'q2': {'z'}}
        out = M.avaliar(rks, rel, ks=(1,))
        # P@1: q1=1, q2=0 -> 0,5; R@1: q1=1, q2=0 -> 0,5; F1 das médias = 0,5 (e não a média dos F1)
        self.assertEqual(out['P@1'], 0.5)
        self.assertEqual(out['R@1'], 0.5)
        self.assertEqual(out['F1@1'], 0.5)
        rks2 = {'q1': ['a', 'b', 'c'], 'q2': ['a', 'b', 'c']}
        rel2 = {'q1': {'a'}, 'q2': {'a', 'b', 'c', 'd'}}
        o2 = M.avaliar(rks2, rel2, ks=(3,))
        p, r = (1 / 3 + 1) / 2, (1 + 3 / 4) / 2                      # P e R médias: 2/3 e 0,875
        self.assertAlmostEqual(o2['F1@3'], 2 * p * r / (p + r))
        media_f1 = (2 * (1 / 3) * 1 / (1 / 3 + 1) + 2 * 1 * 0.75 / 1.75) / 2   # 0,6786: seria a média dos F1
        self.assertNotAlmostEqual(o2['F1@3'], media_f1, places=3)

    def test_consulta_sem_relevante_nao_entra(self):
        out = M.avaliar({'q1': ['a'], 'q2': ['b']}, {'q1': {'a'}, 'q2': set()}, ks=(1,))
        self.assertEqual(out['n_consultas'], 1)


IDS = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
EMP = ['B', 'B', 'B', 'A', 'B', 'B', 'B', 'B']
TEXTOS = ['tubo 25', 'tubo 50', 'luva 25', 'tubo 25', 'cabo 10', 'cap 32', 'joelho 40', 'curva 90']


def okapi(tf, df, n, dl, avgdl, k1, b):
    """BM25 Okapi de um termo (Lin et al. 2021, eq. 2), calculado à mão para conferir a biblioteca."""
    idf = math.log((n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


@unittest.skipIf(bm25 is None, 'precisa de rank-bm25')
class TestBM25(unittest.TestCase):
    """O BM25 é o `BM25Okapi` da biblioteca rank-bm25 0.2.2; os testes conferem o que está em volta dele."""

    def setUp(self):
        self.idx = bm25.BM25(IDS, EMP, TEXTOS, k1=1.2, b=0.75, usar_stopwords=False)

    def test_formula_okapi_conferida_a_mao(self):
        # N = 8; df(tubo) = 3; df(25) = 3; todos os documentos têm 2 termos, L_ave = 2
        s = self.idx.pontuar('tubo 25')
        um = okapi(1, 3, 8, 2, 2, 1.2, 0.75)
        self.assertAlmostEqual(s[0], 2 * um)          # d1: tubo e 25
        self.assertAlmostEqual(s[1], um)              # d2: só tubo
        self.assertAlmostEqual(s[2], um)              # d3: só 25
        self.assertAlmostEqual(s[4], 0.0)             # d5: nenhum termo

    def test_idf_negativo_vira_epsilon_vezes_o_idf_medio(self):
        # comportamento da biblioteca, mantido: termo em mais da metade dos documentos recebe 0,25 x IDF médio
        idx = bm25.BM25(['a', 'b', 'c', 'd'], ['X'] * 4, ['tubo x', 'tubo y', 'tubo z', 'w v'], k1=1.2, b=0.75, usar_stopwords=False)
        n = 4
        idfs = {'tubo': math.log((n - 3 + .5) / (3 + .5)), 'x': math.log((n - 1 + .5) / (1 + .5)), 'y': math.log((n - 1 + .5) / (1 + .5)),
                'z': math.log((n - 1 + .5) / (1 + .5)), 'w': math.log((n - 1 + .5) / (1 + .5)), 'v': math.log((n - 1 + .5) / (1 + .5))}
        self.assertLess(idfs['tubo'], 0)
        eps = 0.25 * (sum(idfs.values()) / len(idfs))
        esperado = eps * 1 * (1.2 + 1) / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / 2))   # dl = 2, L_ave = 2
        self.assertAlmostEqual(idx.pontuar('tubo')[0], esperado)

    def test_termo_repetido_na_consulta_soma_de_novo(self):
        # comportamento da biblioteca, mantido
        for a, b in zip(self.idx.pontuar('tubo tubo'), self.idx.pontuar('tubo')):
            self.assertAlmostEqual(a, 2 * b)

    def test_ranking_exclui_empresa_da_consulta_e_score_zero(self):
        ids, scores = self.idx.ranquear('tubo 25', 'A')
        self.assertEqual(ids, ['d1', 'd2', 'd3'])          # d4 é da empresa A (consulta); d2 e d3 empatam, vale a ordem do pool
        ids, _ = self.idx.ranquear('xyz', 'A')
        self.assertEqual(ids, [])

    def test_ajuste_k1_b_equivale_a_novo_indice(self):
        self.idx.ajustar(2.0, 0.5)
        novo = bm25.BM25(IDS, EMP, TEXTOS, k1=2.0, b=0.5, usar_stopwords=False)
        for a, b in zip(self.idx.pontuar('tubo 50'), novo.pontuar('tubo 50')):
            self.assertAlmostEqual(a, b)

    def test_palavras_funcionais_com_acento(self):
        # achado: "não", "até", "também" não eram filtradas por causa do acento
        self.assertEqual(bm25.tokenizar('Tê Esgoto não é para até 25 mm também com de você são tubo'),
                         ['esgoto', '25', 'mm', 'tubo'])
        self.assertIn('nao', bm25.tokenizar('nao tubo', usar_stopwords=False))

    def test_comprimento_normaliza(self):
        idx = bm25.BM25(['a', 'b', 'c', 'd', 'e', 'f', 'g'], ['X'] * 7,
                        ['tubo', 'tubo luva luva luva', 'cap', 'p1', 'p2', 'p3', 'p4'], k1=1.2, b=0.75, usar_stopwords=False)
        s = idx.pontuar('tubo')
        self.assertGreater(s[0], s[1])          # mesmo tf, documento menor pontua mais


class TestTempoEBloqueio(unittest.TestCase):
    def test_cronometro(self):
        c = tempo.Cronometro()
        c.setup(time.sleep, 0.01)
        for _ in range(3):
            c.consulta(time.sleep, 0.002)
        r = c.resumo()
        self.assertGreaterEqual(r['Setup_ms'], 9)
        self.assertAlmostEqual(r['Total_ms'], r['Setup_ms'] + r['Consulta_ms'])            # Total = Setup + Consulta
        self.assertAlmostEqual(r['Total_conjunto_ms'], r['Setup_ms'] + 3 * r['Consulta_ms'])

    def test_setup_repetido_comeca_do_zero_e_devolve_cinco_tempos(self):
        construidos = []
        obj, tempos = tempo.medir_setup_repetido(lambda: construidos.append(1) or len(construidos), n_rep=5, liberar=lambda: None)
        self.assertEqual(len(tempos), 5)
        self.assertEqual(obj, 5)
        self.assertEqual(len(construidos), 5)

    def test_teste_bloqueado(self):
        with self.assertRaises(dados.TesteBloqueado):
            dados.carregar('teste')

    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_validacao_coerente(self):
        consultas, cands, rel = dados.carregar('validacao')
        self.assertTrue(set(rel) <= {c['id'] for c in consultas})
        self.assertNotIn('tipo', consultas[0])        # o tipo não chega às técnicas
        self.assertTrue(consultas[0]['texto_b'].startswith('[COL]'))      # formato B também para as consultas


if __name__ == '__main__':
    unittest.main()
