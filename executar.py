"""Executa as etapas do pipeline, uma por comando, na ordem do README.

  python executar.py preprocessar      1. dados/bruto/ -> dados/base.csv
  (aqui entra o gabarito: dados/gabarito/ e dados/treino/, feitos por você)
  python executar.py bm25              2. grade do BM25 na validação -> dados/resultados/bm25_escolha.json
  python executar.py tokens            3. comprimento máximo em tokens (treino e validação)
  python executar.py formato           4. Bi-Encoder zero-shot nos formatos A e B; escolhe o formato
  python executar.py treinar           5. Bi-Encoder MNR e Cosine e Cross-Encoder (retoma de onde parou)
  python executar.py hibridas          6. escolhe o K das 4 híbridas na validação
  python executar.py teste --confirmo-teste   7. as 9 variantes no teste (uma vez) e o comparativo
  python executar.py comparar          reescreve o comparativo a partir do que já está em saida/
  python executar.py dispersao         desvio padrão e variância da AP por consulta (teste)

Tudo o que é gerado vai para `saida/` (ou `--saida`). Etapas concluídas não são refeitas; para continuar uma etapa
interrompida, rode o mesmo comando de novo. O teste só abre com --confirmo-teste, depois da validação completa.
"""
import argparse
import os
import sys

RAIZ = os.path.abspath(os.path.dirname(__file__))
for _d in ('1_preprocessamento', '2_modelos', '3_avaliacao', '.'):
    sys.path.insert(0, os.path.join(RAIZ, _d))

import config as C

HIBRIDAS = ('bm25', 'bi_zero_shot', 'bi_cosine', 'bi_mnr')
TESTE_BI = (('zero_shot', 'bi_zero_shot'), ('cosine', 'bi_cosine'), ('mnr', 'bi_mnr'))


def _exigir_gpu(args):
    import torch
    if not torch.cuda.is_available() and not args.aceitar_cpu:
        raise SystemExit('sem GPU: o treino não termina em CPU e os tempos só valem em GPU (use --aceitar-cpu para insistir)')


def _parametros(saida):
    import decisoes as D
    f = D.exigir(saida, 'formato')
    return {'formato': f, 'max_seq': D.exigir(saida, f'max_seq_{f}'), 'max_len': D.exigir(saida, f'max_len_{f}')}


def preprocessar(args):
    import preprocessar as P
    P.main()


def bm25(args):
    import avaliar_bm25 as AB
    AB.grade_validacao()


def tokens(args):
    import decisoes as D
    import medir_tokens
    os.makedirs(args.saida, exist_ok=True)
    d = D.ler(args.saida)
    if all(k in d for k in ('max_seq_A', 'max_len_A', 'max_seq_B', 'max_len_B')):
        print('já medido:', {k: v for k, v in d.items() if k.startswith('max_')})
    else:
        medir_tokens.medir(args.saida)


def formato(args):
    import biencoder_gpu as BG
    import decisoes as D
    import tempo
    _exigir_gpu(args)
    maps = {}
    for f in 'AB':
        m = BG.zero_shot(f, args.saida, None, max_seq=D.exigir(args.saida, f'max_seq_{f}'))
        maps[f] = m['MAP']
        tempo.liberar_gpu()
    D.registrar(args.saida, 'formato', D.escolher_formato(maps))
    print('MAP de validação:', maps, '| formato escolhido:', D.exigir(args.saida, 'formato'))


def treinar(args):
    import biencoder_gpu as BG
    import biencoder_nucleo as BN
    import crossencoder_gpu as CG
    import crossencoder_nucleo as X
    import tempo
    _exigir_gpu(args)
    p = _parametros(args.saida)
    for perda in ('mnr', 'cosine'):
        for cfg in BN.configuracoes(perda):
            BG.treinar(cfg, p['formato'], args.saida, None, max_seq=p['max_seq'])
            tempo.liberar_gpu()
    for cfg in X.configuracoes():
        CG.treinar(cfg, p['formato'], args.saida, None, max_len=p['max_len'])
        tempo.liberar_gpu()
    print('Bi-Encoder (MNR e Cosine) e Cross-Encoder concluídos')


def hibridas(args):
    import hibrida_gpu as HG
    import tempo
    _exigir_gpu(args)
    p = _parametros(args.saida)
    for tipo in HIBRIDAS:
        HG.validar(tipo, args.saida, None, p['formato'], p['formato'], p['max_seq'], p['max_len'])
        tempo.liberar_gpu()


def _validacao_completa(saida):
    """Falta alguma etapa da validação? Devolve a lista do que falta."""
    import decisoes as D
    import hibrida_nucleo as H
    import retomada as R
    faltam = []
    if not os.path.exists(os.path.join(C.DB, 'resultados', 'bm25_escolha.json')):
        faltam.append('bm25')
    d = D.ler(saida)
    if 'formato' not in d:
        return faltam + ['formato (rode tokens e formato)']
    for prefixo in ('bi_mnr_', 'bi_cosine_', 'ce_'):
        try:
            H.melhor_execucao(saida, prefixo)
        except FileNotFoundError:
            faltam.append(prefixo.rstrip('_'))
    for tipo in HIBRIDAS:
        if not R.concluido(os.path.join(saida, f'hibrida_{tipo}_validacao')):
            faltam.append(f'hibrida_{tipo}')
    return faltam


def teste(args):
    import avaliar_bm25 as AB
    import biencoder_gpu as BG
    import comparar
    import crossencoder_gpu as CG
    import decisoes as D
    import hibrida_gpu as HG
    import hibrida_nucleo as H
    import tempo
    if not args.confirmo_teste:
        raise SystemExit('o teste só abre com --confirmo-teste: ele é avaliado uma única vez, depois de todas as escolhas da validação')
    faltam = _validacao_completa(args.saida)
    if faltam:
        raise SystemExit(f'validação incompleta, faltam: {faltam}')
    D.registrar(args.saida, 'teste_liberado', True)
    _exigir_gpu(args)
    p = _parametros(args.saida)

    def caminho_bi(tipo):
        if tipo == 'zero_shot':
            return C.MODELO_BI_ENCODER
        return H.melhor_execucao(args.saida, f'bi_{tipo}_')[0]

    AB.final_teste(args.saida, None)
    tempo.liberar_gpu()
    for tipo, nome in TESTE_BI:
        BG.avaliar_teste(caminho_bi(tipo), p['formato'], args.saida, None, p['max_seq'], nome)
        tempo.liberar_gpu()
    CG.avaliar_teste(H.melhor_execucao(args.saida, 'ce_')[0], p['formato'], args.saida, None, p['max_len'], 'ce_completo')
    tempo.liberar_gpu()
    for tipo in HIBRIDAS:
        HG.avaliar_teste(tipo, args.saida, None, p['formato'], p['formato'], p['max_seq'], p['max_len'])
        tempo.liberar_gpu()
    comparar.gerar(args.saida)
    print(open(os.path.join(args.saida, 'comparativo_teste.md'), encoding='utf-8').read())


def comparar_cmd(args):
    import comparar
    comparar.gerar(args.saida)
    print(open(os.path.join(args.saida, 'comparativo_teste.md'), encoding='utf-8').read())


def dispersao(args):
    import dispersao as DP
    print(DP.gravar(args.saida, DP.calcular(args.saida)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('etapa', choices=['preprocessar', 'bm25', 'tokens', 'formato', 'treinar', 'hibridas', 'teste', 'comparar', 'dispersao'])
    ap.add_argument('--saida', default=os.path.join(RAIZ, 'saida'), help='pasta de saída (padrão: saida/)')
    ap.add_argument('--confirmo-teste', action='store_true', help='abre o conjunto de teste (etapa final)')
    ap.add_argument('--aceitar-cpu', action='store_true', help='permite rodar sem GPU (lento; tempos sem valor)')
    args = ap.parse_args()
    args.saida = os.path.abspath(args.saida)
    os.makedirs(args.saida, exist_ok=True)
    {'preprocessar': preprocessar, 'bm25': bm25, 'tokens': tokens, 'formato': formato, 'treinar': treinar,
     'hibridas': hibridas, 'teste': teste, 'comparar': comparar_cmd, 'dispersao': dispersao}[args.etapa](args)


if __name__ == '__main__':
    main()
