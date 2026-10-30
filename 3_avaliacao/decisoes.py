"""Decisões tomadas na VALIDAÇÃO, registradas em um arquivo. Sem torch: testável localmente.

Guarda, de forma atômica e dentro da pasta persistente, o que cada passo do notebook decidiu (comprimento máximo por
formato, formato A ou B, ...), para que as sessões seguintes leiam o mesmo valor em vez de decidir de novo. Só aceita
decisões sobre treino e validação; o teste nunca entra aqui.
"""
import os

import retomada as R

ARQUIVO = 'decisoes.json'
GRADE_COMPRIMENTO = (128, 192, 256, 512)       # piso 128 (padrão da especificação); sobe só se o percentil 99 passar


def caminho(persistente):
    return os.path.join(persistente, ARQUIVO)


def ler(persistente):
    return R.ler_json(caminho(persistente), {})


def registrar(persistente, chave, valor, sobrescrever=False):
    d = ler(persistente)
    if chave in d and d[chave] != valor and not sobrescrever:
        raise RuntimeError(f'decisão {chave!r} já registrada ({d[chave]!r}); mudar exige sobrescrever=True e justificativa')
    d[chave] = valor
    R.gravar_json(caminho(persistente), d)
    return d


def exigir(persistente, chave):
    d = ler(persistente)
    if chave not in d:
        raise RuntimeError(f'decisão {chave!r} ainda não registrada: rode o passo anterior do notebook')
    return d[chave]


def comprimento_para(p99, grade=GRADE_COMPRIMENTO):
    """Menor valor da grade (piso 128) que cobre o percentil 99; se nem 512 cobre, devolve 512 (limite do BERT)."""
    for g in grade:
        if g >= p99:
            return g
    return grade[-1]


def escolher_formato(map_por_formato):
    """Formato com maior MAP de validação do zero-shot; empate vai para A (o mais simples)."""
    return max(sorted(map_por_formato), key=lambda f: (round(map_por_formato[f], 10)))
