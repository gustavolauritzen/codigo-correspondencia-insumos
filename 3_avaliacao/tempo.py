"""Medição de tempo igual para todas as técnicas.

As três colunas abaixo e o protocolo de repetição (Vitório et al. 2025, §4.2.4,
cinco repetições com a média reportada):
  Setup (ms)    = custo pago uma vez antes da primeira consulta: carregar o(s) modelo(s), tokenizar/codificar/indexar o pool
                  do conjunto, inclusive o do primeiro estágio nas híbridas. Média de N_REP repetições, cada uma a partir
                  do zero (a repetição anterior é liberada).
  Consulta (ms) = custo de UMA consulta, do texto ao ranking final (nas híbridas, primeiro estágio + Cross-Encoder).
                  Média sobre todas as consultas do conjunto, cada uma medida uma vez (as consultas reais já são
                  medidas na própria execução).
  Total (ms)    = Setup + Consulta ("TOTAL (SETUP + CONSULTA)").
Informação adicional, fora da tabela: Total do conjunto = Setup + soma de todas as consultas.
Em GPU, sincroniza antes de ler o relógio.
"""
import gc
import time

N_REP = 5


def sincronizar():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def liberar_gpu():
    """Libera a memória da GPU entre repetições (modelos Large ocupam ~1,3 GB cada)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def agora():
    sincronizar()
    return time.perf_counter()


def medir_setup_repetido(construir, n_rep=N_REP, liberar=liberar_gpu):
    """Executa `construir()` n_rep vezes, cada uma do zero, e devolve (o objeto da última repetição, lista de tempos em ms).
    O objeto anterior é descartado e a memória liberada ANTES de cronometrar a repetição seguinte."""
    tempos, obj = [], None
    for _ in range(n_rep):
        obj = None
        liberar()
        t0 = agora()
        obj = construir()
        tempos.append((agora() - t0) * 1000)
    return obj, tempos


class Cronometro:
    """Cronometra as consultas (e, se preciso, um Setup avulso). O Setup oficial vem de `medir_setup_repetido`."""

    def __init__(self):
        self.setup_ms = 0.0
        self.consultas_ms = []

    def agora(self):
        return agora()

    def setup(self, funcao, *a, **kw):
        t0 = self.agora()
        r = funcao(*a, **kw)
        self.setup_ms += (self.agora() - t0) * 1000
        return r

    def consulta(self, funcao, *a, **kw):
        t0 = self.agora()
        r = funcao(*a, **kw)
        self.consultas_ms.append((self.agora() - t0) * 1000)
        return r

    def resumo(self, setup_ms=None):
        n = len(self.consultas_ms)
        soma = sum(self.consultas_ms)
        consulta = soma / n if n else 0.0
        setup = self.setup_ms if setup_ms is None else setup_ms
        return {'Setup_ms': setup, 'Consulta_ms': consulta, 'Total_ms': setup + consulta, 'Total_conjunto_ms': setup + soma}


def resumir(setup_ms, consultas_ms):
    """Mesmo cálculo a partir de uma lista de tempos (usado ao reler o CSV de progresso)."""
    n = len(consultas_ms)
    consulta = sum(consultas_ms) / n if n else 0.0
    return {'Setup_ms': setup_ms, 'Consulta_ms': consulta, 'Total_ms': setup_ms + consulta,
            'Total_conjunto_ms': setup_ms + sum(consultas_ms)}
