"""Testes das decisões da validação e da medição de tokens."""
import os
import sys
import tempfile
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TEM_DADOS = all(os.path.exists(os.path.join(_RAIZ, 'dados', *p)) for p in (
    ('base.csv',), ('gabarito', 'validacao', 'gabarito.csv'), ('treino', 'pares_positivos.csv')))
SEM_DADOS = 'precisa de dados/ (base, gabarito e pares de treino, feitos por quem executa)'
import decisoes as D
import medir_tokens as MT


class TestDecisoes(unittest.TestCase):
    def test_comprimento_para(self):
        self.assertEqual(D.comprimento_para(40), 128)           # piso
        self.assertEqual(D.comprimento_para(128), 128)
        self.assertEqual(D.comprimento_para(129), 192)
        self.assertEqual(D.comprimento_para(300), 512)
        self.assertEqual(D.comprimento_para(900), 512)          # limite do BERT

    def test_registrar_nao_muda_sem_pedir(self):
        with tempfile.TemporaryDirectory() as d:
            D.registrar(d, 'formato', 'A')
            D.registrar(d, 'formato', 'A')                      # mesmo valor: ok
            with self.assertRaises(RuntimeError):
                D.registrar(d, 'formato', 'B')
            D.registrar(d, 'formato', 'B', sobrescrever=True)
            self.assertEqual(D.exigir(d, 'formato'), 'B')
            with self.assertRaises(RuntimeError):
                D.exigir(d, 'inexistente')

    def test_escolher_formato(self):
        self.assertEqual(D.escolher_formato({'A': 0.50, 'B': 0.45}), 'A')
        self.assertEqual(D.escolher_formato({'A': 0.40, 'B': 0.45}), 'B')
        self.assertEqual(D.escolher_formato({'A': 0.5, 'B': 0.5}), 'A')


class TestMedirTokens(unittest.TestCase):
    def test_estatisticas(self):
        e = MT.estatisticas(list(range(1, 101)), corte=90)
        self.assertEqual(e['n'], 100)
        self.assertAlmostEqual(e['p50'], 50.5)
        self.assertAlmostEqual(e['fracao_acima_do_corte'], 0.10)
        self.assertEqual(e['max'], 100)

    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_nao_le_o_teste(self):
        import ast
        fonte = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2_modelos', 'medir_tokens.py'), encoding='utf-8').read()
        arvore = ast.parse(fonte)
        literais = [n.value for n in ast.walk(arvore) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertNotIn('teste', literais)
        self.assertNotIn('liberar_teste', fonte)

    @unittest.skipUnless(TEM_DADOS, SEM_DADOS)
    def test_textos_de_treino_e_validacao(self):
        soltos, pares = MT.textos_treino_e_validacao('A')
        self.assertGreater(len(soltos), 10000)
        self.assertGreater(len(pares), 50000)


if __name__ == '__main__':
    unittest.main()
