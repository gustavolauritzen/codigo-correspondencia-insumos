"""Testes de salvar progresso e retomar."""
import os
import sys
import tempfile
import unittest

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import metricas as M
import retomada as R
import tempo


class TestArquivos(unittest.TestCase):
    def test_escrita_atomica_e_json(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, 'a', 'x.json')
            R.gravar_json(c, {'a': 1})
            self.assertEqual(R.ler_json(c), {'a': 1})
            self.assertFalse(os.path.exists(c + '.tmp'))
            self.assertIsNone(R.ler_json(os.path.join(d, 'nada.json')))

    def test_copia_verificada_substitui_so_depois_de_conferir(self):
        with tempfile.TemporaryDirectory() as d:
            o, p = os.path.join(d, 'o'), os.path.join(d, 'p', 'run')
            os.makedirs(os.path.join(o, 'sub'))
            open(os.path.join(o, 'sub', 'm.bin'), 'wb').write(b'1234')
            R.copiar_verificado(o, p)
            self.assertEqual(open(os.path.join(p, 'sub', 'm.bin'), 'rb').read(), b'1234')
            open(os.path.join(o, 'sub', 'm.bin'), 'wb').write(b'novo!')
            R.copiar_verificado(o, p)
            self.assertEqual(open(os.path.join(p, 'sub', 'm.bin'), 'rb').read(), b'novo!')
            self.assertEqual([x for x in os.listdir(os.path.dirname(p))], ['run'])      # sem .tmp nem .antigo

    def test_restaurar_so_se_o_local_faltar(self):
        with tempfile.TemporaryDirectory() as d:
            p, l = os.path.join(d, 'p'), os.path.join(d, 'l')
            os.makedirs(p)
            open(os.path.join(p, 'f'), 'w').write('x')
            self.assertTrue(R.restaurar_se_faltar(p, l))
            open(os.path.join(l, 'f'), 'w').write('local mais novo')
            self.assertFalse(R.restaurar_se_faltar(p, l))
            self.assertEqual(open(os.path.join(l, 'f')).read(), 'local mais novo')

    def test_concluido(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(R.concluido(d))
            R.gravar_json(os.path.join(d, 'concluido.json'), {})
            self.assertTrue(R.concluido(d))


class TestConsultas(unittest.TestCase):
    CONS = [{'id': f'q{i}', 'empresa': 'A'} for i in range(5)]
    REL = {f'q{i}': {'d1'} for i in range(5)}

    def ranquear(self, q):
        return ['d0', 'd1'] if q['id'] in ('q1', 'q3') else ['d1', 'd0']

    def test_retomada_equivale_a_execucao_continua(self):
        with tempfile.TemporaryDirectory() as d:
            reg = R.RegistroConsultas(os.path.join(d, 'c.csv'))
            cron = tempo.Cronometro()
            # sessão 1 cai depois de 2 consultas (simulado: só as 2 primeiras)
            R.executar_consultas(reg, self.CONS[:2], self.REL, self.ranquear, cron)
            self.assertEqual(reg.feitas(), {'q0', 'q1'})
            with self.assertRaises(RuntimeError):
                R.agregar_registro(reg, self.CONS, self.REL)           # incompleto: não calcula métricas
            # sessão 2 continua: pula as feitas
            rodou, ja = R.executar_consultas(reg, self.CONS, self.REL, self.ranquear, tempo.Cronometro())
            self.assertEqual((rodou, ja), (3, 2))
            retomado = R.agregar_registro(reg, self.CONS, self.REL, setup_ms=10.0)
            direto = M.avaliar({q['id']: self.ranquear(q) for q in self.CONS}, self.REL)
            for k in ('P@1', 'R@1', 'F1@1', 'P@5', 'MRR', 'MAP', 'MRP'):
                self.assertAlmostEqual(retomado[k], direto[k], msg=k)
            self.assertEqual(retomado['Setup_ms'], 10.0)
            self.assertAlmostEqual(retomado['Total_ms'], 10.0 + retomado['Consulta_ms'])
            self.assertAlmostEqual(retomado['Total_conjunto_ms'], 10.0 + retomado['Consulta_ms'] * 5)

    def test_nao_regrava_consulta_ja_feita(self):
        with tempfile.TemporaryDirectory() as d:
            reg = R.RegistroConsultas(os.path.join(d, 'c.csv'))
            R.executar_consultas(reg, self.CONS, self.REL, self.ranquear, tempo.Cronometro())
            R.executar_consultas(reg, self.CONS, self.REL, self.ranquear, tempo.Cronometro())
            self.assertEqual(len(reg.linhas()), 5)

    def test_persistir_chamado(self):
        with tempfile.TemporaryDirectory() as d:
            reg = R.RegistroConsultas(os.path.join(d, 'c.csv'))
            chamadas = []
            R.executar_consultas(reg, self.CONS, self.REL, self.ranquear, tempo.Cronometro(), persistir=lambda: chamadas.append(1), cada=2)
            self.assertGreaterEqual(len(chamadas), 2)

    def test_agregar_igual_a_avaliar_em_varias_consultas(self):
        rks = {'q1': ['a', 'b', 'c'], 'q2': ['x', 'a', 'z', 'b']}
        rel = {'q1': {'a'}, 'q2': {'a', 'b', 'c', 'd'}}
        linhas = [dict(M.por_consulta(rks[q], rel[q]), query_id=q) for q in rks]
        ag, av = M.agregar(linhas), M.avaliar(rks, rel)
        for k in av:
            self.assertAlmostEqual(ag[k], av[k], msg=k)


if __name__ == '__main__':
    unittest.main()
