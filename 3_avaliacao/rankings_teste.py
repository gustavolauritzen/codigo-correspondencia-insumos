"""Rankings do teste e Preparação das 9 variantes numa mesma sessão. Roda em GPU, depois do teste (`executar.py teste`).

Usa os mesmos modelos e o mesmo código do teste e não escolhe nada: grava, consulta a consulta (retoma se cair),
{'query_id', 'ids', 'scores'} em `saida/<nome>/ranking.jsonl`, para a R@K do primeiro estágio, a avaliação humana do
primeiro colocado e a análise de erros.

  primeiro_estagio_teste(tipo)     os 200 primeiros do 1º estágio (bm25, bi_zero_shot, bi_cosine, bi_mnr)
  hibrida_top_teste(tipo, k)       os K do 1º estágio com a pontuação do Cross-Encoder (K da validação)
  ce_amostra_teste()               Cross-Encoder sobre todos os candidatos numa amostra fixa de 100 consultas
  ce_completo_teste()              Cross-Encoder sobre todos os candidatos nas consultas do teste (top-10)
  setups_mesma_sessao()            Preparação das 9 variantes numa só sessão: aquecimento não cronometrado e
                                   5 repetições de cada, do zero (protocolo de tempo.medir_setup_repetido)
"""
import os
import random
import sys
import time

for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d)))
import config as C
import dados
import retomada as R

TOPO_PRIMEIRO_ESTAGIO = 200
N_AMOSTRA_CE, SEMENTE_AMOSTRA, TOPO_CE = 100, 42, 10
TIPOS_PRIMEIRO_ESTAGIO = ('bm25', 'bi_zero_shot', 'bi_cosine', 'bi_mnr')


def caminho_modelo(tipo, saida):
    """Modelo do 1º estágio: HF (zero-shot) ou melhor checkpoint da validação ('bi_mnr', 'bi_cosine', 'ce')."""
    import hibrida_nucleo as H
    if tipo == 'bi_zero_shot':
        return C.MODELO_BI_ENCODER
    return H.melhor_execucao(saida, 'ce_' if tipo == 'ce' else f'{tipo}_')[0]


def _registrar_rankings(nome, consultas, funcao, saida, cada=200, meta=None):
    import biencoder_gpu as BG
    pl, pp = BG.pastas(saida, None, nome)
    if R.concluido(pl):
        print(f'{nome}: já concluído')
        return pl
    reg = R.RegistroJsonl(os.path.join(pl, 'ranking.jsonl'))
    feitas = reg.ler()
    for n, q in enumerate(consultas, 1):
        if q['id'] in feitas:
            continue
        ids, sc = funcao(q)
        reg.acrescentar({'query_id': q['id'], 'ids': [str(i) for i in ids], 'scores': [float(s) for s in sc]})
        if n % cada == 0:
            print(f'  {nome}: {n}/{len(consultas)}', flush=True)
    feitas = reg.ler()
    faltam = [q['id'] for q in consultas if q['id'] not in feitas]
    if faltam:
        raise RuntimeError(f'{nome}: faltam {len(faltam)} consultas')
    R.gravar_json(os.path.join(pl, 'meta.json'), {**(meta or {}), 'n_consultas': len(consultas), 'gpu': BG.nome_gpu()})
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'n_consultas': len(consultas)})
    print(f'{nome}: concluído ({len(consultas)} consultas)')
    return pl


def primeiro_estagio_teste(tipo, saida, formato, max_seq, topo=TOPO_PRIMEIRO_ESTAGIO):
    import biencoder_gpu as BG
    import hibrida_gpu as HG
    import tempo
    consultas, cands, _ = dados.carregar('teste', liberar_teste=True)
    if tipo == 'bm25':
        idx = HG.PrimeiroEstagioBM25(cands).idx
        funcao = lambda q: idx.ranquear(q['texto'], q['empresa'], topo=topo)
    else:
        rec = BG.criar_recuperador(caminho_modelo(tipo, saida), formato, cands, max_seq)
        funcao = lambda q: rec.ranquear(q, topo=topo)
    pl = _registrar_rankings(f'rank_teste_{tipo}', consultas, funcao, saida, meta={'tipo': tipo, 'topo': topo})
    tempo.liberar_gpu()
    return pl


def hibrida_top_teste(tipo, k, saida, formato, max_seq, max_len):
    """Os K do 1º estágio (em ordem) com a pontuação do Cross-Encoder; a híbrida reordena por ela."""
    import hibrida_gpu as HG
    import tempo
    consultas, cands, _ = dados.carregar('teste', liberar_teste=True)
    hib = HG.Hibrida(tipo, cands, saida, None, formato, formato, max_seq, max_len)
    pl = _registrar_rankings(f'rank_teste_hibrida_{tipo}', consultas, lambda q: hib.top_com_scores(q, k), saida,
                             meta={'tipo': tipo, 'K': k, 'ordem': 'primeiro estágio; scores = Cross-Encoder'})
    tempo.liberar_gpu()
    return pl


def amostra_ids(consultas, n=N_AMOSTRA_CE, semente=SEMENTE_AMOSTRA):
    """Amostra aleatória fixa das consultas do teste (a da avaliação humana do primeiro colocado)."""
    return sorted(random.Random(semente).sample(sorted(q['id'] for q in consultas), n))


def _ce_pool_inteiro(nome, alvo, cands, saida, formato, max_len, topo, meta):
    import crossencoder_gpu as CG
    import tempo
    rr = CG.criar_reranqueador_completo(caminho_modelo('ce', saida), formato, cands, max_len)

    def f(q):
        o, s = rr.reranquear(q, [d for d in cands if d['empresa'] != q['empresa']])
        return o[:topo], s[:topo]
    pl = _registrar_rankings(nome, alvo, f, saida, cada=10, meta=meta)
    tempo.liberar_gpu()
    return pl


def ce_amostra_teste(saida, formato, max_len, topo=TOPO_CE):
    consultas, cands, _ = dados.carregar('teste', liberar_teste=True)
    ids = set(amostra_ids(consultas))
    return _ce_pool_inteiro('rank_teste_ce_amostra', [q for q in consultas if q['id'] in ids], cands, saida, formato,
                            max_len, topo, {'tipo': 'ce', 'amostra': sorted(ids), 'semente': SEMENTE_AMOSTRA, 'topo': topo})


def ce_completo_teste(saida, formato, max_len, topo=TOPO_CE):
    consultas, cands, _ = dados.carregar('teste', liberar_teste=True)
    return _ce_pool_inteiro('rank_teste_ce_completo', consultas, cands, saida, formato, max_len, topo,
                            {'tipo': 'ce', 'pool': 'completo', 'topo': topo})


def construtores(saida, formato, max_seq, max_len):
    """(nome, construir(candidatos)) das 9 variantes, como no teste."""
    import biencoder_gpu as BG
    import crossencoder_gpu as CG
    import hibrida_gpu as HG
    m = {t: caminho_modelo(t, saida) for t in ('bi_zero_shot', 'bi_cosine', 'bi_mnr', 'ce')}
    hib = lambda t: (lambda c: HG.Hibrida(t, c, saida, None, formato, formato, max_seq, max_len))
    return [
        ('BM25', lambda c: HG.PrimeiroEstagioBM25(c)),
        ('Bi-Encoder zero-shot', lambda c: BG.criar_recuperador(m['bi_zero_shot'], formato, c, max_seq)),
        ('Bi-Encoder Cosine', lambda c: BG.criar_recuperador(m['bi_cosine'], formato, c, max_seq)),
        ('Bi-Encoder MNR', lambda c: BG.criar_recuperador(m['bi_mnr'], formato, c, max_seq)),
        ('Cross-Encoder', lambda c: CG.criar_reranqueador_completo(m['ce'], formato, c, max_len)),
        ('Híbrida BM25', hib('bm25')),
        ('Híbrida Bi-Encoder zero-shot', hib('bi_zero_shot')),
        ('Híbrida Bi-Encoder Cosine', hib('bi_cosine')),
        ('Híbrida Bi-Encoder MNR', hib('bi_mnr')),
    ]


def setups_mesma_sessao(saida, formato, max_seq, max_len):
    """Preparação das 9 variantes numa só sessão. Um aquecimento não cronometrado constrói cada variante uma vez, para
    que nenhuma pague sozinha a primeira leitura dos pesos do disco. Grava só no fim (se cair, recomeça do zero)."""
    import biencoder_gpu as BG
    import tempo
    arq = os.path.join(saida, 'setups_mesma_sessao.json')
    if os.path.exists(arq):
        print('já medido:', arq)
        return R.ler_json(arq)
    _, cands, _ = dados.carregar('teste', liberar_teste=True)
    lista = construtores(saida, formato, max_seq, max_len)
    for _, construir in lista:
        obj = construir(cands)
        del obj
        tempo.liberar_gpu()
    out = {'gpu': BG.nome_gpu(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S'), 'protocolo': 'aquecimento + 5 repetições',
           'variantes': {}}
    for nome, construir in lista:
        obj, tempos = tempo.medir_setup_repetido(lambda: construir(cands))
        del obj
        tempo.liberar_gpu()
        out['variantes'][nome] = {'Setup_ms': sum(tempos) / len(tempos), 'repeticoes_ms': tempos}
        print(f'{nome}: Preparação {sum(tempos) / len(tempos):.0f} ms', flush=True)
    if BG.nome_gpu() != out['gpu']:
        raise RuntimeError('a GPU mudou durante a medição')
    R.gravar_json(arq, out)
    return out
