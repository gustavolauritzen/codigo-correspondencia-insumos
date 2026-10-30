"""Híbridas: primeiro estágio -> Cross-Encoder dos K melhores. Roda em GPU.

As quatro variantes: BM25 -> CE, Bi zero-shot -> CE, Bi Cosine -> CE, Bi MNR -> CE. Usam os melhores checkpoints
escolhidos na VALIDAÇÃO (Bi Cosine e MNR, Cross-Encoder) e o BM25 com a escolha da validação. O K (20, 50, 100 ou 200) é
escolhido na validação por MAP; no teste o K já vem fixado. Salvar progresso e retomar como no resto (retomada.py): os
scores da validação são gravados consulta a consulta e o teste usa o protocolo único de execucao_teste.py (Setup = carregar
os dois estágios e preparar o pool, média de 5 repetições; Consulta = primeiro estágio + Cross-Encoder).

Uso:
  python hibrida_gpu.py validar --tipo bi_mnr --local saida
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import argparse
import json
import os

import numpy as np

import biencoder_gpu as BG
import bm25 as BM
import config as C
import crossencoder_gpu as CG
import crossencoder_nucleo as X
import dados
import hibrida_nucleo as H
import retomada as R

ESCOLHA_BM25 = os.path.join(C.DB, 'resultados', 'bm25_escolha.json')


def garantir_local(local, persistente, prefixo):
    """Traz da pasta persistente para o disco local as execuções `prefixo*` que faltam (sessão nova)."""
    if persistente and os.path.isdir(persistente):
        for nome in sorted(os.listdir(persistente)):
            if nome.startswith(prefixo):
                R.restaurar_se_faltar(os.path.join(persistente, nome), os.path.join(local, nome))


class PrimeiroEstagioBM25:
    def __init__(self, cands, escolha=ESCOLHA_BM25):
        e = R.ler_json(escolha)
        self.idx = BM.BM25([d['id'] for d in cands], [d['empresa'] for d in cands], [d['texto'] for d in cands],
                           k1=e['k1'], b=e['b'], usar_stopwords=e['stopwords'])
        self.descricao = {'tipo': 'bm25', **e}

    def ranquear(self, q, topo):
        return self.idx.ranquear(q['texto'], q['empresa'], topo=topo)[0]


class PrimeiroEstagioBi:
    def __init__(self, modelo, formato, cands, max_seq, descricao):
        self.rec = BG.criar_recuperador(modelo, formato, cands, max_seq)
        self.descricao = descricao

    def ranquear(self, q, topo):
        return self.rec.ranquear(q, topo=topo)[0]


def localizar_modelos(tipo, local, persistente, formato_bi, formato_ce):
    """Caminhos dos modelos de cada estágio: (modelo do 1º estágio ou None, pasta do melhor Cross-Encoder, descrições)."""
    garantir_local(local, persistente, 'ce_')
    ce_path, ce_info = H.melhor_execucao(local, 'ce_')
    if tipo == 'bm25':
        return None, ce_path, {'ce': ce_info}
    if tipo == 'bi_zero_shot':
        return C.MODELO_BI_ENCODER, ce_path, {'ce': ce_info, 'bi': 'zero-shot'}
    perda = tipo.split('_')[1]
    garantir_local(local, persistente, f'bi_{perda}_')
    bi_path, bi_info = H.melhor_execucao(local, f'bi_{perda}_')
    return bi_path, ce_path, {'ce': ce_info, 'bi': bi_info}


class Hibrida:
    def __init__(self, tipo, cands, local, persistente, formato_bi='A', formato_ce='A', max_seq=None, max_len=None):
        from biencoder_nucleo import MAX_SEQ_PROVISORIO
        max_seq = max_seq or MAX_SEQ_PROVISORIO
        max_len = max_len or X.MAX_LEN_PROVISORIO
        bi_path, ce_path, info = localizar_modelos(tipo, local, persistente, formato_bi, formato_ce)
        self.tipo, self.info, self.ce_path = tipo, info, ce_path
        self.pool = {d['id']: d for d in cands}
        self.estagio1 = (PrimeiroEstagioBM25(cands) if tipo == 'bm25'
                         else PrimeiroEstagioBi(bi_path, formato_bi, cands, max_seq, info))
        self.rr = CG.criar_reranqueador_completo(ce_path, formato_ce, cands, max_len)      # pool tokenizado no Setup

    def top_com_scores(self, q, k):
        """Os K primeiros do 1º estágio (em ordem) e os scores do Cross-Encoder nessa mesma ordem."""
        ids = self.estagio1.ranquear(q, k)
        if not ids:
            return [], []
        campo = CG.CAMPO[self.rr.formato]
        sc = CG.pontuar(self.rr.modelo, self.rr.cache, q[campo], [self.pool[i][campo] for i in ids], self.rr.max_len,
                        self.rr.dispositivo, self.rr.torch)
        return ids, [float(s) for s in sc]

    def ranquear(self, q, k):
        ids, sc = self.top_com_scores(q, k)
        return H.reordenar(ids, sc, k)


def validar(tipo, local, persistente=None, formato_bi='A', formato_ce='A', max_seq=None, max_len=None, persistir_a_cada=50):
    """Escolhe o K na VALIDAÇÃO (todas as consultas com relevante). Grava os scores consulta a consulta e retoma."""
    nome = f'hibrida_{tipo}_validacao'
    pl, pp = BG.pastas(local, persistente, nome)
    if R.concluido(pl):
        print(f'{nome}: já concluído')
        return R.ler_json(os.path.join(pl, 'escolha.json'))
    consultas, cands, rel = dados.carregar('validacao')
    hib = Hibrida(tipo, cands, local, persistente, formato_bi, formato_ce, max_seq, max_len)
    reg = R.RegistroJsonl(os.path.join(pl, 'scores.jsonl'))
    feitas = reg.ler()
    kmax = max(H.K_GRADE)
    alvo = [q for q in consultas if rel.get(q['id'])]
    for n, q in enumerate(alvo, 1):
        if q['id'] in feitas:
            continue
        ids, sc = hib.top_com_scores(q, kmax)
        reg.acrescentar({'query_id': q['id'], 'ids': ids, 'scores': sc})
        if n % persistir_a_cada == 0 and pp:
            R.copiar_arquivo_verificado(reg.caminho, os.path.join(pp, 'scores.jsonl'))
            print(f'  {nome}: {n}/{len(alvo)} consultas', flush=True)
    todas = reg.ler()
    top = {q['id']: todas[q['id']]['ids'] for q in alvo}
    scs = {q['id']: todas[q['id']]['scores'] for q in alvo}
    por_k = H.metricas_por_k(top, scs, rel)
    k = H.escolher_k(por_k)
    escolha = {'tipo': tipo, 'K': k, 'MAP_validacao': por_k[k]['MAP'], 'MRR_validacao': por_k[k]['MRR'],
               'por_K': {str(kk): {'MAP': v['MAP'], 'MRR': v['MRR']} for kk, v in por_k.items()},
               'formato_bi': formato_bi, 'formato_ce': formato_ce,
               'modelos': {kk: (v if isinstance(v, str) else {a: b for a, b in v.items() if a in ('melhor_epoca', 'MAP_validacao', 'config')})
                           for kk, v in hib.info.items()}}
    R.gravar_json(os.path.join(pl, 'escolha.json'), escolha)
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'K': k})
    BG.persistir(pl, pp)
    print(f'{nome}: K = {k} (MAP validação {por_k[k]["MAP"]:.4f})')
    return escolha


def avaliar_teste(tipo, local, persistente=None, formato_bi='A', formato_ce='A', max_seq=None, max_len=None):
    """ETAPA FINAL da híbrida `tipo`, com o K fixado na validação e o protocolo único de execucao_teste.py."""
    import execucao_teste
    pl, pp = BG.pastas(local, persistente, f'hibrida_{tipo}_validacao')
    if not R.concluido(pl):
        raise RuntimeError(f'rode `validar --tipo {tipo}` antes: o K vem da validação')
    escolha = R.ler_json(os.path.join(pl, 'escolha.json'))
    k = escolha['K']
    return execucao_teste.avaliar_teste_tecnica(
        f'teste_hibrida_{tipo}', lambda cands: Hibrida(tipo, cands, local, persistente, formato_bi, formato_ce, max_seq, max_len),
        lambda hib, q: hib.ranquear(q, k), local, persistente,
        meta={'tecnica': H.NOMES[tipo], 'K': k, 'formato_bi': formato_bi, 'formato_ce': formato_ce, 'max_seq': max_seq, 'max_len': max_len, 'escolha': escolha})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('acao', choices=['validar'])
    ap.add_argument('--tipo', choices=H.TIPOS, required=True)
    ap.add_argument('--formato', choices=['A', 'B'], default='A')
    ap.add_argument('--local', required=True)
    ap.add_argument('--persistente')
    a = ap.parse_args()
    validar(a.tipo, a.local, a.persistente, a.formato, a.formato)
