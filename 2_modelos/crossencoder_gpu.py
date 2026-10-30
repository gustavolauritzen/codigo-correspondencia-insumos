"""Cross-Encoder, parte com modelo. Roda em GPU.

monoBERT (Lin et al. 2021, §3.2) sobre o BERTimbau Large (o mesmo BERT Large do Bi-Encoder; `modelos.py` confere a
arquitetura e a versão fixada). Perda: entropia cruzada binária (positivo = equivalente). Treino com os pares de
`dados/treino/`; a cada época mede MAP/MRR na VALIDAÇÃO (amostra fixa de consultas contra o pool inteiro) e
guarda o melhor checkpoint; o teste só é lido por `avaliar_teste`.

SALVA PROGRESSO E RETOMA (retomada.py): estado completo ao fim de cada época, copiado para a pasta persistente com conferência;
no teste, cada consulta é gravada na hora. Rodar o mesmo comando em outra sessão continua de onde parou; execução com
`concluido.json` é pulada.

Uso (pelo `executar.py`, ou direto):
  python crossencoder_gpu.py grade --formato A --local saida
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import argparse
import os
import random
import shutil

import numpy as np

import biencoder_nucleo as BN
import config as C
import crossencoder_nucleo as X
import dados
import metricas as M
import modelos
import retomada as R
import tempo
from biencoder_gpu import nome_gpu, pastas, persistir

CAMPO = {'A': 'texto', 'B': 'texto_b'}


def carregar_ce(nome_ou_dir):
    """BERTimbau Large com cabeça de 1 saída. Identificador do Hugging Face usa a revisão fixada; pasta local é conferida."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    eh_pasta = os.path.isdir(nome_ou_dir)
    modelos.conferir_identificadores()
    kw = {} if eh_pasta else {'revision': C.MODELO_CROSS_ENCODER_REV}
    tok = AutoTokenizer.from_pretrained(nome_ou_dir, **kw)
    modelo = AutoModelForSequenceClassification.from_pretrained(nome_ou_dir, num_labels=1, **kw)
    modelos.conferir_arquitetura(modelo.config)
    return tok, modelo


class Tokenizados:
    """Cache de ids por texto (sem tokens especiais): a consulta e cada candidato são tokenizados uma vez só."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def ids(self, texto):
        if texto not in self.cache:
            self.cache[texto] = self.tok(texto, add_special_tokens=False)['input_ids']
        return self.cache[texto]


def _tensores(pares_ids, pad_id, torch, dispositivo):
    n = max(len(i) for i, _ in pares_ids)
    ids = torch.full((len(pares_ids), n), pad_id, dtype=torch.long)
    tt = torch.zeros((len(pares_ids), n), dtype=torch.long)
    mask = torch.zeros((len(pares_ids), n), dtype=torch.long)
    for k, (i, t) in enumerate(pares_ids):
        ids[k, :len(i)] = torch.tensor(i)
        tt[k, :len(t)] = torch.tensor(t)
        mask[k, :len(i)] = 1
    return ids.to(dispositivo), tt.to(dispositivo), mask.to(dispositivo)


def pontuar(modelo, cache, texto_consulta, textos_candidatos, max_len, dispositivo, torch, batch=X.BATCH_PONTUACAO):
    """Scores (um por candidato, na ordem de entrada) do par (consulta, candidato)."""
    tok = cache.tok
    q = cache.ids(texto_consulta)
    pares = [X.montar_par(q, cache.ids(t), tok.cls_token_id, tok.sep_token_id, max_len) for t in textos_candidatos]
    scores = np.zeros(len(pares), dtype=np.float32)
    modelo.eval()
    with torch.no_grad():
        for idx in X.lotes_por_comprimento([len(p[0]) for p in pares], batch):
            ids, tt, mask = _tensores([pares[i] for i in idx], tok.pad_token_id, torch, dispositivo)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dispositivo == 'cuda'):
                logits = modelo(input_ids=ids, token_type_ids=tt, attention_mask=mask).logits.squeeze(-1)
            scores[idx] = logits.float().cpu().numpy()
    return scores


def avaliar_validacao(modelo, cache, formato, max_len, dispositivo, torch):
    """MAP/MRR etc. na validação: amostra fixa de consultas contra TODOS os candidatos de outras empresas do pool."""
    consultas, cands, rel = dados.carregar('validacao')
    campo = CAMPO[formato]
    rankings = {}
    for q in X.amostra_validacao(consultas, rel):
        outros = [d for d in cands if d['empresa'] != q['empresa']]
        sc = pontuar(modelo, cache, q[campo], [d[campo] for d in outros], max_len, dispositivo, torch)
        rankings[q['id']] = X.ranquear_por_scores([d['id'] for d in outros], sc)
    return M.avaliar(rankings, rel)


def _salvar_estado(pl, modelo, tok, otim, agenda, escala, meta, torch):
    tmp = os.path.join(pl, 'estado.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    modelo.save_pretrained(os.path.join(tmp, 'modelo'))
    tok.save_pretrained(os.path.join(tmp, 'modelo'))
    torch.save({'otimizador': otim.state_dict(), 'agenda': agenda.state_dict(), 'escala': escala.state_dict(),
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               os.path.join(tmp, 'otimizador.pt'))
    R.gravar_json(os.path.join(tmp, 'meta.json'), meta)
    final = os.path.join(pl, 'estado')
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)


def treinar(cfg, formato, local, persistente=None, max_len=X.MAX_LEN_PROVISORIO):
    """Treina uma configuração (taxa de aprendizado) com avaliação na validação a cada época; retoma do último estado."""
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup
    nome = f"ce_lr{cfg['lr']:.0e}_fmt{formato}"
    pl, pp = pastas(local, persistente, nome)
    if R.concluido(pl):
        print(f'{nome}: já concluído, pulando')
        return R.ler_json(os.path.join(pl, 'historico_validacao.json'))
    random.seed(cfg['semente'])
    np.random.seed(cfg['semente'])
    torch.manual_seed(cfg['semente'])
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = R.ler_json(os.path.join(pl, 'estado', 'meta.json'))
    tok, modelo = carregar_ce(os.path.join(pl, 'estado', 'modelo') if meta else C.MODELO_CROSS_ENCODER)
    modelo.to(dispositivo)
    cache = Tokenizados(tok)
    pares = X.ler_pares_ce(formato)
    otim = torch.optim.AdamW(modelo.parameters(), lr=cfg['lr'])
    passos = -(-len(pares) // cfg['batch'])
    total = passos * cfg['epocas']
    agenda = get_linear_schedule_with_warmup(otim, int(cfg['warmup'] * total), total)
    escala = torch.amp.GradScaler('cuda', enabled=dispositivo == 'cuda')
    historico, melhor, epoca_feita = [], None, 0
    if meta:
        est = torch.load(os.path.join(pl, 'estado', 'otimizador.pt'), map_location=dispositivo, weights_only=False)
        otim.load_state_dict(est['otimizador'])
        agenda.load_state_dict(est['agenda'])
        escala.load_state_dict(est['escala'])
        torch.set_rng_state(est['rng_cpu'].cpu())
        if est.get('rng_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in est['rng_cuda']])
        historico, melhor, epoca_feita = meta['historico'], meta['melhor'], meta['epoca_feita']
        print(f'{nome}: retomando depois da época {epoca_feita}', flush=True)
    for epoca in range(epoca_feita + 1, cfg['epocas'] + 1):
        modelo.train()
        soma, n = 0.0, 0
        for lote in X.lotes_ce(pares, cfg['batch'], cfg['semente'] + epoca):
            pid = [X.montar_par(cache.ids(a), cache.ids(b), tok.cls_token_id, tok.sep_token_id, max_len) for a, b, _ in lote]
            ids, tt, mask = _tensores(pid, tok.pad_token_id, torch, dispositivo)
            y = torch.tensor([r for *_, r in lote], dtype=torch.float32, device=dispositivo)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dispositivo == 'cuda'):
                logits = modelo(input_ids=ids, token_type_ids=tt, attention_mask=mask).logits.squeeze(-1)
            perda = F.binary_cross_entropy_with_logits(logits.float(), y)
            otim.zero_grad()
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            agenda.step()
            soma, n = soma + float(perda), n + 1
        m = avaliar_validacao(modelo, cache, formato, max_len, dispositivo, torch)
        m.update({'epoca': epoca, 'perda_media': soma / max(n, 1), 'gpu': nome_gpu()})
        historico.append(m)
        print(f"{nome} época {epoca}: perda {m['perda_media']:.4f} MAP {m['MAP']:.4f} MRR {m['MRR']:.4f}", flush=True)
        if melhor is None or BN.escolher_melhor([melhor, m])['epoca'] == m['epoca']:
            melhor = m
            tmp = os.path.join(pl, 'melhor.tmp')
            shutil.rmtree(tmp, ignore_errors=True)
            modelo.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            shutil.rmtree(os.path.join(pl, 'melhor'), ignore_errors=True)
            os.rename(tmp, os.path.join(pl, 'melhor'))
        R.gravar_json(os.path.join(pl, 'historico_validacao.json'),
                      {'config': cfg, 'formato': formato, 'max_len': max_len, 'historico': historico, 'melhor_epoca': melhor['epoca']})
        _salvar_estado(pl, modelo, tok, otim, agenda, escala, {'epoca_feita': epoca, 'historico': historico, 'melhor': melhor}, torch)
        persistir(pl, pp)
    shutil.rmtree(os.path.join(pl, 'estado'), ignore_errors=True)
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'melhor_epoca': melhor['epoca'], 'MAP_validacao': melhor['MAP'],
                                                         'MRR_validacao': melhor['MRR'], 'config': cfg, 'formato': formato})
    persistir(pl, pp)
    return R.ler_json(os.path.join(pl, 'historico_validacao.json'))


def executar_grade(formato, local, persistente=None):
    for cfg in X.configuracoes():
        treinar(cfg, formato, local, persistente)


class Reranqueador:
    """Segundo estágio das híbridas: pontua só os candidatos que o primeiro estágio entrega."""

    def __init__(self, pasta_modelo, formato, max_len=X.MAX_LEN_PROVISORIO):
        import torch
        self.torch = torch
        self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.formato, self.max_len = formato, max_len
        tok, modelo = carregar_ce(pasta_modelo)
        self.modelo = modelo.to(self.dispositivo)
        self.modelo.eval()
        self.cache = Tokenizados(tok)

    def reranquear(self, consulta, candidatos):
        """`candidatos`: lista de dicionários do pool (ordem do primeiro estágio). Devolve (ids reordenados, scores)."""
        campo = CAMPO[self.formato]
        sc = pontuar(self.modelo, self.cache, consulta[campo], [d[campo] for d in candidatos], self.max_len, self.dispositivo, self.torch)
        ids = [d['id'] for d in candidatos]
        ordem = np.lexsort((np.arange(len(sc)), -sc))
        return [ids[i] for i in ordem], [float(sc[i]) for i in ordem]


def criar_reranqueador_completo(pasta_modelo, formato, cands, max_len=X.MAX_LEN_PROVISORIO):
    """Setup do Cross-Encoder de conjunto completo: carrega o modelo e tokeniza o pool uma vez."""
    rr = Reranqueador(pasta_modelo, formato, max_len)
    campo = CAMPO[formato]
    for d in cands:
        rr.cache.ids(d[campo])
    rr.pool = cands
    return rr


def avaliar_teste(pasta_modelo, formato, local, persistente=None, max_len=X.MAX_LEN_PROVISORIO, nome='ce_completo'):
    """ETAPA FINAL: Cross-Encoder sobre TODO o pool de outras empresas, uma vez, com o protocolo único de execucao_teste.py."""
    import execucao_teste
    return execucao_teste.avaliar_teste_tecnica(
        f'teste_{nome}', lambda cands: criar_reranqueador_completo(pasta_modelo, formato, cands, max_len),
        lambda rr, q: rr.reranquear(q, [d for d in rr.pool if d['empresa'] != q['empresa']])[0], local, persistente,
        meta={'tecnica': nome, 'modelo': str(pasta_modelo), 'formato': formato, 'max_len': max_len}, cada=50)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('acao', choices=['grade'])
    ap.add_argument('--formato', choices=['A', 'B'], default='A')
    ap.add_argument('--local', required=True)
    ap.add_argument('--persistente')
    a = ap.parse_args()
    executar_grade(a.formato, a.local, a.persistente)
