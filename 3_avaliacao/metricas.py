"""Métricas de ranqueamento.

Um ranking é a lista de ids de documentos em ordem decrescente de score; documentos fora da lista contam como
não recuperados. `relevantes` mapeia consulta -> conjunto de ids relevantes (só consultas com ao menos um
relevante são avaliadas). O F1-Score@k é a média das F1 de cada consulta (Opitz e Burst, 2019), calculada a partir
dos acertos gravados por consulta em `f1_por_consulta.py`.
"""
KS = (1, 5, 10)


def _acertos_ate(ranking, rel, k):
    return sum(1 for d in ranking[:k] if d in rel)


def precisao_em_k(ranking, rel, k):
    return _acertos_ate(ranking, rel, k) / k


def revocacao_em_k(ranking, rel, k):
    return _acertos_ate(ranking, rel, k) / len(rel)


def rr(ranking, rel):
    for pos, d in enumerate(ranking, 1):
        if d in rel:
            return 1.0 / pos
    return 0.0


def ap(ranking, rel):
    acertos, soma = 0, 0.0
    for pos, d in enumerate(ranking, 1):
        if d in rel:
            acertos += 1
            soma += acertos / pos
    return soma / len(rel)


def r_precisao(ranking, rel):
    """MRP por consulta: precisão no corte R = número de relevantes."""
    r = len(rel)
    return _acertos_ate(ranking, rel, r) / r


def avaliar(rankings, relevantes, ks=KS):
    """rankings: consulta -> lista de ids. Devolve dicionário de métricas médias e o número de consultas."""
    consultas = [q for q in relevantes if relevantes[q] and q in rankings]
    if not consultas:
        raise ValueError('nenhuma consulta com relevante e ranking')
    n = len(consultas)
    out = {'n_consultas': n}
    for k in ks:
        p = sum(precisao_em_k(rankings[q], relevantes[q], k) for q in consultas) / n
        r = sum(revocacao_em_k(rankings[q], relevantes[q], k) for q in consultas) / n
        out[f'P@{k}'], out[f'R@{k}'] = p, r
    out['MRR'] = sum(rr(rankings[q], relevantes[q]) for q in consultas) / n
    out['MAP'] = sum(ap(rankings[q], relevantes[q]) for q in consultas) / n
    out['MRP'] = sum(r_precisao(rankings[q], relevantes[q]) for q in consultas) / n
    return out


def por_consulta(ranking, rel, ks=KS):
    """Resultado de UMA consulta, em campos que se somam depois (permite gravar consulta a consulta e retomar)."""
    linha = {'n_rel': len(rel), 'ap': ap(ranking, rel), 'rr': rr(ranking, rel), 'rp': r_precisao(ranking, rel)}
    for k in ks:
        linha[f'h{k}'] = _acertos_ate(ranking, rel, k)
    return linha


def agregar(linhas, ks=KS):
    """Mesmas métricas de `avaliar`, a partir das linhas de `por_consulta` (lidas do disco): P@k = média de h_k / k,
    R@k = média de h_k / n_rel."""
    n = len(linhas)
    out = {'n_consultas': n}
    for k in ks:
        p = sum(float(r[f'h{k}']) / k for r in linhas) / n
        r_ = sum(float(r[f'h{k}']) / float(r['n_rel']) for r in linhas) / n
        out[f'P@{k}'], out[f'R@{k}'] = p, r_
    out['MRR'] = sum(float(r['rr']) for r in linhas) / n
    out['MAP'] = sum(float(r['ap']) for r in linhas) / n
    out['MRP'] = sum(float(r['rp']) for r in linhas) / n
    return out
