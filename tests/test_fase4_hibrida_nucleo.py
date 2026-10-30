"""Testes da parte sem GPU das híbridas."""
import os
import sys
import tempfile
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import hibrida_nucleo as H
import retomada as R


class TestReordenar(unittest.TestCase):
    def test_so_os_k_primeiros_e_reordenados(self):
        self.assertEqual(H.reordenar(['a', 'b', 'c', 'd'], [0.1, 0.9, 0.5, 0.99], 3), ['b', 'c', 'a'])      # d fica de fora

    def test_empate_mantem_ordem_do_primeiro_estagio(self):
        self.assertEqual(H.reordenar(['a', 'b', 'c'], [0.5, 0.5, 0.5], 3), ['a', 'b', 'c'])

    def test_relevante_fora_do_k_nao_e_recuperado(self):
        top = {'q': ['x', 'y', 'rel']}
        sc = {'q': [0.1, 0.2, 0.9]}
        por_k = H.metricas_por_k(top, sc, {'q': {'rel'}}, ks=(2, 3))
        self.assertEqual(por_k[2]['MRR'], 0.0)
        self.assertEqual(por_k[3]['MRR'], 1.0)


class TestEscolhaK(unittest.TestCase):
    def test_map_depois_mrr_depois_menor_k(self):
        por_k = {20: {'MAP': 0.5, 'MRR': 0.7}, 50: {'MAP': 0.6, 'MRR': 0.7}, 100: {'MAP': 0.6, 'MRR': 0.7}, 200: {'MAP': 0.55, 'MRR': 0.9}}
        self.assertEqual(H.escolher_k(por_k), 50)

    def test_grade_e_tipos(self):
        self.assertEqual(H.K_GRADE, (20, 50, 100, 200))
        self.assertEqual(len(H.TIPOS), 4)
        self.assertEqual(set(H.NOMES), set(H.TIPOS))


class TestMelhorExecucao(unittest.TestCase):
    def test_escolhe_maior_map_entre_concluidas(self):
        with tempfile.TemporaryDirectory() as d:
            for nome, mp, concl in (('bi_mnr_lr3e-05_fmtA', 0.60, True), ('bi_mnr_lr2e-05_fmtA', 0.65, True), ('bi_mnr_lr5e-05_fmtA', 0.90, False),
                                    ('bi_cosine_lr3e-05_fmtA', 0.99, True)):
                os.makedirs(os.path.join(d, nome))
                if concl:
                    R.gravar_json(os.path.join(d, nome, 'concluido.json'), {'MAP_validacao': mp, 'MRR_validacao': 0.7})
            caminho, c = H.melhor_execucao(d, 'bi_mnr_')
            self.assertTrue(caminho.endswith('bi_mnr_lr2e-05_fmtA/melhor'))
            self.assertEqual(c['MAP_validacao'], 0.65)
            with self.assertRaises(FileNotFoundError):
                H.melhor_execucao(d, 'ce_')


class TestJsonl(unittest.TestCase):
    def test_ignora_linha_cortada(self):
        with tempfile.TemporaryDirectory() as d:
            reg = R.RegistroJsonl(os.path.join(d, 'x.jsonl'))
            reg.acrescentar({'query_id': 'q1', 'v': 1})
            with open(reg.caminho, 'a') as f:
                f.write('{"query_id": "q2", "v"')         # sessão caiu no meio da escrita
            self.assertEqual(set(reg.ler()), {'q1'})


if __name__ == '__main__':
    unittest.main()
