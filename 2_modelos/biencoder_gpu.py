"""Bi-Encoder, parte com modelo. Roda em GPU.

Modelo: BERTimbau Large com pooling por média (config.MODELO_BI_ENCODER). Variantes: zero-shot, Cosine (regressão do
cosseno contra o rótulo 1/0, Reimers e Gurevych 2019) e MNR (negativos do lote, Henderson et al. §4.4).
Treino em laço próprio (sem `fit` do sentence-transformers, cuja API muda entre versões): AdamW, aquecimento
linear de 10%, fp16, avaliação na VALIDAÇÃO a cada época (MAP, desempate MRR) e checkpoint só da melhor época.

SALVA PROGRESSO E RETOMA (retomada.py): o treino grava o estado completo ao fim de cada época e o copia para a pasta
persistente com conferência de tamanho e MD5; a avaliação do teste grava cada consulta na hora. Rodar o mesmo
comando de novo, em outra sessão, continua de onde parou; uma execução com `concluido.json` é pulada, nunca refeita.

Travas: nada concluído é sobrescrito; o trabalho acontece no disco local
(`local`) e só é copiado para a pasta persistente (`persistente`); o teste só é lido por `avaliar_teste`.

Uso (pelo `executar.py`, ou direto; `--persistente` é opcional):
  python biencoder_gpu.py zero_shot --formato A --local saida
  python biencoder_gpu.py treinar --perda mnr --lr 3e-5 --formato A --local saida
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

import biencoder_nucleo as N
import config as C
import dados
import metricas as M
import retomada as R
import tempo


# ── utilidades ───────────────────────────────────────────────────────────────
def nome_gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def pastas(local, persistente, nome):
    pl = os.path.join(local, nome)
    pp = os.path.join(persistente, nome) if persistente else None
    if pp:
        R.restaurar_se_faltar(pp, pl)
    os.makedirs(pl, exist_ok=True)
    return pl, pp


def persistir(pl, pp):
    if pp:
        R.copiar_verificado(pl, pp)


def carregar_st(nome_ou_dir, max_seq=N.MAX_SEQ_PROVISORIO):
    """Carrega o Bi-Encoder conferindo que é o BERTimbau Large. O identificador do Hugging Face usa a revisão fixada em
    config; pastas locais (checkpoints nossos) são conferidas pela arquitetura."""
    from sentence_transformers import SentenceTransformer
    import modelos
    eh_pasta = os.path.isdir(nome_ou_dir)
    modelos.conferir_identificadores()
    st = SentenceTransformer(nome_ou_dir, **({} if eh_pasta else {'revision': C.MODELO_BI_ENCODER_REV}))
    modelos.conferir_arquitetura(st[0].auto_model.config)
    st.max_seq_length = max_seq
    return st


def codificar(st, textos, batch=64):
    return st.encode(textos, batch_size=batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)


def _campo(formato):
    return {'A': 'texto', 'B': 'texto_b'}[formato]


def avaliar_validacao(st, formato):
    consultas, cands, rel = dados.carregar('validacao')
    campo = _campo(formato)
    emb_c = codificar(st, [d[campo] for d in cands])
    emb_q = codificar(st, [q[campo] for q in consultas])
    return N.avaliar_vetores(emb_q, consultas, emb_c, cands, rel)


class Recuperador:
    """Primeiro estágio denso, reutilizado pelas híbridas: pool codificado uma vez (Setup), ranking por consulta."""

    def __init__(self, st, cands, formato='A'):
        self.st, self.cands, self.formato = st, cands, formato
        campo = _campo(formato)
        emb = codificar(st, [d[campo] for d in cands])
        self.emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        self.ids = np.array([d['id'] for d in cands])
        self.empresas = np.array([d['empresa'] for d in cands])

    def ranquear(self, consulta, empresa=None, topo=None):
        """`consulta` = dicionário da consulta (usa o campo do formato escolhido)."""
        q = codificar(self.st, [consulta[_campo(self.formato)]], batch=1)[0]
        q = q / max(np.linalg.norm(q), 1e-12)
        s = self.emb @ q
        permitido = np.nonzero(self.empresas != (empresa or consulta['empresa']))[0]
        ordem = permitido[np.lexsort((permitido, -s[permitido]))]
        if topo is not None:
            ordem = ordem[:topo]
        return list(self.ids[ordem]), [float(s[i]) for i in ordem]


# ── perdas ───────────────────────────────────────────────────────────────────
def _emb(st, textos, dispositivo):
    feats = st.tokenize(textos)
    feats = {k: v.to(dispositivo) for k, v in feats.items() if hasattr(v, 'to')}
    return st(feats)['sentence_embedding']


def _perda_cosine(st, lote, dispositivo, torch):
    import torch.nn.functional as F
    a = _emb(st, [x[0] for x in lote], dispositivo)
    b = _emb(st, [x[1] for x in lote], dispositivo)
    y = torch.tensor([x[2] for x in lote], dtype=torch.float32, device=dispositivo)
    return F.mse_loss(F.cosine_similarity(a, b).float(), y)


def _perda_mnr(st, lote, dispositivo, torch):
    import torch.nn.functional as F
    a = F.normalize(_emb(st, [x[0] for x in lote], dispositivo), dim=-1)
    b = F.normalize(_emb(st, [x[1] for x in lote], dispositivo), dim=-1)
    scores = N.ESCALA_MNR * (a @ b.T).float()
    return F.cross_entropy(scores, torch.arange(len(lote), device=dispositivo))


# ── treino com retomada ──────────────────────────────────────────────────────
def _salvar_estado(pl, st, otim, agenda, escala, meta, torch):
    tmp = os.path.join(pl, 'estado.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    st.save(os.path.join(tmp, 'modelo'))
    torch.save({'otimizador': otim.state_dict(), 'agenda': agenda.state_dict(), 'escala': escala.state_dict(),
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               os.path.join(tmp, 'otimizador.pt'))
    R.gravar_json(os.path.join(tmp, 'meta.json'), meta)
    final = os.path.join(pl, 'estado')
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)


def treinar(cfg, formato, local, persistente=None, max_seq=N.MAX_SEQ_PROVISORIO):
    """Treina uma configuração (perda, lr) com avaliação na validação a cada época. Retoma do último estado salvo."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    nome = f"bi_{cfg['perda']}_lr{cfg['lr']:.0e}_fmt{formato}"
    pl, pp = pastas(local, persistente, nome)
    if R.concluido(pl):
        print(f'{nome}: já concluído, pulando')
        return R.ler_json(os.path.join(pl, 'historico_validacao.json'))
    random.seed(cfg['semente'])
    np.random.seed(cfg['semente'])
    torch.manual_seed(cfg['semente'])
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = R.ler_json(os.path.join(pl, 'estado', 'meta.json'))
    fonte = os.path.join(pl, 'estado', 'modelo') if meta else C.MODELO_BI_ENCODER
    st = carregar_st(fonte, max_seq).to(dispositivo)
    pos, neg = N.ler_pares_treino(formato)
    otim = torch.optim.AdamW(st.parameters(), lr=cfg['lr'])
    passos = (len(N.lotes_sem_repeticao(pos, cfg['batch'], cfg['semente'] + 1)) if cfg['perda'] == 'mnr'
              else -(-(len(pos) + len(neg)) // cfg['batch']))
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
        print(f'{nome}: retomando depois da época {epoca_feita} (histórico de {len(historico)} épocas)', flush=True)
    for epoca in range(epoca_feita + 1, cfg['epocas'] + 1):
        lotes = (N.lotes_sem_repeticao(pos, cfg['batch'], cfg['semente'] + epoca) if cfg['perda'] == 'mnr'
                 else N.lotes_com_rotulo(pos, neg, cfg['batch'], cfg['semente'] + epoca))
        st.train()
        soma = 0.0
        for lote in lotes:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dispositivo == 'cuda'):
                perda = (_perda_mnr if cfg['perda'] == 'mnr' else _perda_cosine)(st, lote, dispositivo, torch)
            otim.zero_grad()
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            agenda.step()
            soma += float(perda)
        st.eval()
        with torch.no_grad():
            m = avaliar_validacao(st, formato)
        m.update({'epoca': epoca, 'perda_media': soma / max(len(lotes), 1), 'gpu': nome_gpu()})
        historico.append(m)
        print(f"{nome} época {epoca}: perda {m['perda_media']:.4f} MAP {m['MAP']:.4f} MRR {m['MRR']:.4f}", flush=True)
        if melhor is None or N.escolher_melhor([melhor, m])['epoca'] == m['epoca']:
            melhor = m
            tmp = os.path.join(pl, 'melhor.tmp')
            shutil.rmtree(tmp, ignore_errors=True)
            st.save(tmp)
            shutil.rmtree(os.path.join(pl, 'melhor'), ignore_errors=True)
            os.rename(tmp, os.path.join(pl, 'melhor'))
        R.gravar_json(os.path.join(pl, 'historico_validacao.json'),
                      {'config': cfg, 'formato': formato, 'max_seq': max_seq, 'historico': historico, 'melhor_epoca': melhor['epoca']})
        _salvar_estado(pl, st, otim, agenda, escala, {'epoca_feita': epoca, 'historico': historico, 'melhor': melhor}, torch)
        persistir(pl, pp)                    # copia e confere antes de seguir para a próxima época
    shutil.rmtree(os.path.join(pl, 'estado'), ignore_errors=True)     # o estado só serve para retomar
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'melhor_epoca': melhor['epoca'], 'MAP_validacao': melhor['MAP'],
                                                         'MRR_validacao': melhor['MRR'], 'config': cfg, 'formato': formato})
    persistir(pl, pp)
    return R.ler_json(os.path.join(pl, 'historico_validacao.json'))


def zero_shot(formato, local, persistente=None, max_seq=N.MAX_SEQ_PROVISORIO):
    nome = f'bi_zero_shot_fmt{formato}'
    pl, pp = pastas(local, persistente, nome)
    arq = os.path.join(pl, 'validacao.json')
    if R.concluido(pl):
        print(f'{nome}: já concluído, pulando')
        return R.ler_json(arq)['metricas']
    st = carregar_st(C.MODELO_BI_ENCODER, max_seq)
    m = avaliar_validacao(st, formato)
    R.gravar_json(arq, {'formato': formato, 'max_seq': max_seq, 'metricas': m, 'gpu': nome_gpu()})
    R.gravar_json(os.path.join(pl, 'concluido.json'), {'MAP_validacao': m['MAP']})
    persistir(pl, pp)
    print(f"zero-shot formato {formato}: MAP {m['MAP']:.4f} MRR {m['MRR']:.4f}")
    return m


def executar_grade(formato, local, persistente=None, perdas=('mnr', 'cosine')):
    """Roda (e retoma) todas as configurações da validação, uma depois da outra. Pode ser chamada em várias sessões."""
    for perda in perdas:
        for cfg in N.configuracoes(perda):
            treinar(cfg, formato, local, persistente)


# ── teste (etapa final) com retomada ─────────────────────────────────────────
def criar_recuperador(modelo, formato, cands, max_seq=N.MAX_SEQ_PROVISORIO):
    """Setup do Bi-Encoder: carrega o modelo e codifica o pool (é isso que o Setup cronometra)."""
    return Recuperador(carregar_st(modelo, max_seq), cands, formato)


def avaliar_teste(modelo, formato, local, persistente=None, max_seq=N.MAX_SEQ_PROVISORIO, nome='bi'):
    """ETAPA FINAL: avalia no teste uma vez com o protocolo único de execucao_teste.py. `modelo` = nome do HF (zero-shot)
    ou pasta do melhor checkpoint."""
    import execucao_teste
    return execucao_teste.avaliar_teste_tecnica(
        f'teste_{nome}', lambda cands: criar_recuperador(modelo, formato, cands, max_seq),
        lambda rec, q: rec.ranquear(q)[0], local, persistente,
        meta={'tecnica': nome, 'modelo': str(modelo), 'formato': formato, 'max_seq': max_seq})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('acao', choices=['zero_shot', 'treinar', 'grade'])
    ap.add_argument('--formato', choices=['A', 'B'], default='A')
    ap.add_argument('--perda', choices=['cosine', 'mnr'])
    ap.add_argument('--lr', type=float)
    ap.add_argument('--local', required=True)
    ap.add_argument('--persistente')
    a = ap.parse_args()
    if a.acao == 'zero_shot':
        zero_shot(a.formato, a.local, a.persistente)
    elif a.acao == 'grade':
        executar_grade(a.formato, a.local, a.persistente)
    else:
        treinar(next(c for c in N.configuracoes(a.perda) if abs(c['lr'] - a.lr) < 1e-12), a.formato, a.local, a.persistente)
