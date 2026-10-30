"""Execução FINAL no conjunto de teste, idêntica para as 9 variantes.

Uma só rotina mede e salva tudo, para que BM25, Bi-Encoder (3), Cross-Encoder e híbridas (4) tenham o mesmo protocolo:
  1. Setup: `construir(candidatos)` (carrega modelo(s) e prepara o pool) medido cinco vezes, cada uma do zero, média
     (tempo.py; Vitório et al. 2025, §4.2.4). Só na primeira sessão; as seguintes reaproveitam `setup.json`.
  2. Consulta: cada consulta do teste é ranqueada e cronometrada uma vez e gravada na hora (retomada.py).
  3. Métricas P@k, R@k, F1@k (k = 1, 5, 10), MRR, MAP, MRP e Setup/Consulta/Total, relidas do disco quando TODAS as
     consultas existem. Saída: `teste.json` (+ linha para o comparativo).
O teste só é aberto aqui (`liberar_teste=True`), nunca antes.
"""
import os

import dados
import retomada as R
import tempo


def avaliar_teste_tecnica(nome, construir, ranquear, local, persistente=None, meta=None, cada=100):
    """nome: pasta de saída (por exemplo 'teste_bi_mnr'); construir(candidatos) -> objeto pronto;
    ranquear(objeto, consulta) -> lista de ids em ordem; meta: dicionário gravado junto (modelo, parâmetros...)."""
    from biencoder_gpu import nome_gpu, pastas, persistir        # import tardio: não exige torch para importar este módulo
    pl, pp = pastas(local, persistente, nome)
    saida = os.path.join(pl, 'teste.json')
    if R.concluido(pl):
        print(f'{nome}: já concluído')
        return R.ler_json(saida)['metricas']
    consultas, cands, rel = dados.carregar('teste', liberar_teste=True)
    setup = R.ler_json(os.path.join(pl, 'setup.json'))
    if setup is None:
        obj, tempos = tempo.medir_setup_repetido(lambda: construir(cands))
        setup = {'Setup_ms': sum(tempos) / len(tempos), 'repeticoes_ms': tempos, 'gpu': nome_gpu()}
        R.gravar_json(os.path.join(pl, 'setup.json'), setup)
        if pp:                                                   # o Setup medido tem de sobreviver à sessão
            R.copiar_arquivo_verificado(os.path.join(pl, 'setup.json'), os.path.join(pp, 'setup.json'))
        print(f"{nome}: Setup {setup['Setup_ms']:.0f} ms (média de {len(tempos)} repetições)", flush=True)
    else:
        obj = construir(cands)                                   # sessão de retomada: prepara de novo, sem cronometrar
    reg = R.RegistroConsultas(os.path.join(pl, 'consultas.csv'))

    def salvar_csv():
        if pp and os.path.exists(reg.caminho):
            R.copiar_arquivo_verificado(reg.caminho, os.path.join(pp, 'consultas.csv'))

    cron = tempo.Cronometro()
    R.executar_consultas(reg, consultas, rel, lambda q: ranquear(obj, q), cron, gpu=nome_gpu(), rotulo=nome,
                         persistir=salvar_csv, cada=cada)
    out = R.agregar_registro(reg, consultas, rel, setup_ms=setup['Setup_ms'])
    out['gpu_setup'] = setup['gpu']
    R.gravar_json(saida, {'meta': meta or {}, 'metricas': out})
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'MAP': out['MAP']})
    persistir(pl, pp)
    return out
