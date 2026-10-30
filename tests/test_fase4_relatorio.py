"""Testes do relatório e do comparativo das 9 variantes."""
import os
import sys
import tempfile
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import comparar as CP
import relatorio as RL
import retomada as R


def metricas(n=10, f=0.5):
    m = {'n_consultas': n, 'MRR': f, 'MAP': f, 'MRP': f, 'Setup_ms': 1000.0, 'Consulta_ms': 12.5, 'Total_ms': 1012.5}
    for k in (1, 5, 10):
        m.update({f'P@{k}': f, f'R@{k}': f, f'F1@{k}': f})
    return m


class TestRelatorio(unittest.TestCase):
    def test_as_15_colunas_na_ordem(self):
        self.assertEqual(RL.COLUNAS, ['P@1', 'P@5', 'P@10', 'R@1', 'R@5', 'R@10', 'F1@1', 'F1@5', 'F1@10', 'MRR', 'MAP', 'MRP',
                                      'Setup (ms)', 'Consulta (ms)', 'Total (ms)'])
        self.assertEqual(list(RL.linha(metricas())), RL.COLUNAS)

    def test_linha_exige_tudo(self):
        m = metricas()
        del m['Setup_ms']
        with self.assertRaises(KeyError):
            RL.linha(m)

    def test_texto_no_formato_do_relatorio(self):
        t = RL.texto_metricas('BM25', metricas())
        for trecho in ('Avaliação Cross-Company', 'SETUP (carregamento/preparação, uma vez)', 'CONSULTA (por consulta)',
                       'TOTAL (SETUP + CONSULTA)', 'P@1', 'F1@10', 'MRR :', 'MAP :', 'MRP :'):
            self.assertIn(trecho, t)


class TestComparativo(unittest.TestCase):
    def preparar(self, d, variantes):
        for _, sub in variantes:
            p = os.path.join(d, sub)
            R.gravar_json(os.path.join(p, 'teste.json'), {'meta': {}, 'metricas': metricas()})
            R.gravar_json(os.path.join(p, 'concluido.json'), {})

    def test_nove_variantes(self):
        self.assertEqual(len(CP.VARIANTES), 9)
        nomes = [n for n, _ in CP.VARIANTES]
        self.assertEqual(sum(1 for n in nomes if n.startswith('Híbrida')), 4)

    def test_gera_tabela_com_as_nove(self):
        with tempfile.TemporaryDirectory() as d:
            self.preparar(d, CP.VARIANTES)
            linhas, ausentes = CP.gerar(d)
            self.assertEqual(len(linhas), 9)
            self.assertEqual(ausentes, [])
            self.assertTrue(os.path.exists(os.path.join(d, 'comparativo_teste.csv')))
            self.assertTrue(os.path.exists(os.path.join(d, 'teste_bm25_metricas.txt')))

    def test_recusa_se_faltar_variante(self):
        with tempfile.TemporaryDirectory() as d:
            self.preparar(d, CP.VARIANTES[:5])
            with self.assertRaises(RuntimeError):
                CP.gerar(d)
            linhas, ausentes = CP.gerar(d, exigir_todas=False)
            self.assertEqual(len(linhas), 5)
            self.assertEqual(len(ausentes), 4)

    def test_recusa_numero_diferente_de_consultas(self):
        with tempfile.TemporaryDirectory() as d:
            self.preparar(d, CP.VARIANTES)
            R.gravar_json(os.path.join(d, 'teste_bi_mnr', 'teste.json'), {'metricas': metricas(n=9)})
            with self.assertRaises(RuntimeError):
                CP.gerar(d)


if __name__ == '__main__':
    unittest.main()
