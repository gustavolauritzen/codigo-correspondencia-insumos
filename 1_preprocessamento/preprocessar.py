"""Pré-processamento.

1. Lê a coleta bruta em dados/bruto/ e seleciona as empresas acima do corte de itens.
2. Remove nome_insumo, palavras_chave, tamanho e referencia_item.
3. Constrói os textos (variantes A e B) e deduplica, dentro de cada empresa, os itens de texto final
   idêntico (sem diferença de caixa), mantendo o menor id.
4. Grava dados/base.csv, dados/mapa_dedup.csv e dados/estatisticas_preprocessamento.txt.

A divisão em treino, validação e teste e o gabarito são feitos depois, por quem executa (dados/gabarito/LEIA-ME.md).
"""
import os as _os
import sys as _sys
for _d in ('..', '../1_preprocessamento', '../2_modelos', '../3_avaliacao'):
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)))
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import texto as T


def empresas_acima_do_corte():
    with open(os.path.join(C.COLETA_DIR, f'ranking_empresas_{C.PERIODO_RANKING}.csv'),
              encoding='utf-8-sig', newline='') as f:
        return {r['nome_ficticio']: int(r['itens_ordem']) for r in csv.DictReader(f)
                if r['nome_ficticio'] and int(r['itens_ordem']) > C.CORTE_ITENS_ORDEM}


def ler_coleta(empresas):
    """Lê a coleta mantendo só as colunas permitidas (categoria nunca é carregada)."""
    itens = []
    with open(os.path.join(C.COLETA_DIR, 'base_dados_insumos.csv'),
              encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['banco_origem'] in empresas:
                itens.append({c: row[c] for c in C.COLUNAS_MANTIDAS})
    return itens


def sem_descricao_do_produto(item):
    """Sem nome nem descrição após a limpeza (só unidade ou vazio): não descreve um insumo."""
    nome, desc, _ = T.campos_limpos(item)
    return not nome and not desc


def deduplicar(itens):
    """Mantém o menor id de cada (empresa, texto final). Retorna (mantidos, mapa, copias)."""
    grupos = defaultdict(list)
    for it in itens:
        grupos[(it['banco_origem'], T.chave_dedup(it))].append(it)
    mantidos, mapa, copias = [], [], {}
    for g in grupos.values():
        g.sort(key=lambda x: int(x['id']))
        base = g[0]
        mantidos.append(base)
        copias[base['id']] = len(g)
        mapa.extend((x['id'], base['id']) for x in g[1:])
    mantidos.sort(key=lambda x: int(x['id']))
    return mantidos, mapa, copias


def main():
    os.makedirs(C.DB, exist_ok=True)
    acima = empresas_acima_do_corte()
    itens = ler_coleta(set(acima))
    assert not (set(itens[0]) & set(C.COLUNAS_REMOVIDAS)), 'coluna removida carregada'

    # Catálogo mínimo (substitui a justificativa antiga baseada em nome_insumo).
    por_empresa = Counter(i['banco_origem'] for i in itens)
    excluidas = sorted(e for e, n in por_empresa.items() if n < C.MIN_ITENS_CATALOGO)
    selecionadas = {e for e in acima if e not in excluidas}
    itens = [i for i in itens if i['banco_origem'] in selecionadas]

    vazios = [i for i in itens if sem_descricao_do_produto(i)]
    itens = [i for i in itens if not sem_descricao_do_produto(i)]

    mantidos, mapa, copias = deduplicar(itens)

    saida = os.path.join(C.DB, 'base.csv')
    with open(saida, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(list(C.COLUNAS_MANTIDAS) + ['n_copias', 'texto_a', 'texto_b'])
        for i in mantidos:
            w.writerow([i[c] for c in C.COLUNAS_MANTIDAS]
                       + [copias[i['id']], T.texto_a(i, C.EXPANDIR_SIGLAS), T.texto_b(i, C.EXPANDIR_SIGLAS)])
    with open(os.path.join(C.DB, 'mapa_dedup.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id_removido', 'id_mantido'])
        w.writerows(mapa)

    antes = Counter(i['banco_origem'] for i in itens)
    depois = Counter(i['banco_origem'] for i in mantidos)
    linhas = [
        'Pré-processamento',
        f'Empresas acima do corte (itens_ordem > {C.CORTE_ITENS_ORDEM}): {len(acima)}',
        f'Excluídas por catálogo < {C.MIN_ITENS_CATALOGO} itens: {excluidas} ({[por_empresa[e] for e in excluidas]} itens)',
        f'Empresas selecionadas: {len(selecionadas)}',
        f'Itens sem nome nem descrição do produto (descartados antes da deduplicação): {len(vazios)}',
        f'Itens antes da deduplicação: {len(itens)}',
        f'Itens depois da deduplicação: {len(mantidos)} ({len(mapa)} removidos, {len(mapa)/len(itens):.1%})',
        '', 'empresa | itens antes | depois',
    ]
    for e in sorted(selecionadas, key=lambda x: int(x.split()[-1])):
        linhas.append(f'{e} | {antes[e]} | {depois[e]}')
    with open(os.path.join(C.DB, 'estatisticas_preprocessamento.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n')
    print('\n'.join(linhas))


if __name__ == '__main__':
    main()
