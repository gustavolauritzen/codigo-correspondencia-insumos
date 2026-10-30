"""Construção do texto dos insumos.

Especificação:
  - campos usados: nome_item, descricao_item, unidade (nada de categoria);
  - limpeza de ruído comercial por campo (preço, tributo, código/referência
    interna, desconto entre colchetes, percentual, data), sem alterar o
    valor numérico de atributos reais;
  - normalização de número e unidade;
  - variante A: concatenação simples; variante B: [COL] rótulo [VAL] valor.
"""
import re
import unicodedata

# ── limpeza de ruído comercial ───────────────────────────────────────────────
_RUIDOS = [
    ('colchetes',  re.compile(r'\[[^\]]*\]')),
    ('preco',      re.compile(r'\(\s*R\$[^)]*\)')),
    ('tributo',    re.compile(r'Trib\.?\s*Aprox\.?.*$', re.IGNORECASE)),
    ('codigo',     re.compile(r'\bC[OÓ]D\.?:?\s*\d+[\d.]*', re.IGNORECASE)),
    ('referencia', re.compile(r'\bREF\.?:?\s*\d+[\d.]*', re.IGNORECASE)),
    ('percentual', re.compile(r'[+-]?\s*\d+\s*%')),
    ('data',       re.compile(r'\b\d{2}/\d{2}/\d{2,4}\b')),
]


def limpar_ruido(texto):
    for _, padrao in _RUIDOS:
        texto = padrao.sub('', texto)
    return re.sub(r'\s+', ' ', texto).strip()


# ── normalização de número e unidade ────────────────────────────────────────
_LETRA = r'A-Za-zÀ-ÿ'

def normalizar(texto):
    """Minúsculas, sem acento nem sublinhado, espaços únicos (usada pelo BM25 e pelas chaves de comparação)."""
    t = unicodedata.normalize('NFKD', texto.lower().replace('_', ' ').replace('´´', '"').replace('’’', '"'))
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', t).strip()


def normalizar_numeros_unidades(texto):
    """²/³ para 2/3; vírgula decimal para ponto (só entre dígitos); hífen para
    espaço; separa número e letra coladas ("35.0MM", "CA50")."""
    texto = texto.replace('²', '2').replace('³', '3')
    texto = re.sub(r'(?<=\d),(?=\d)', '.', texto)
    texto = texto.replace('-', ' ')
    texto = re.sub(rf'(\d)([{_LETRA}])', r'\1 \2', texto)
    texto = re.sub(rf'([{_LETRA}])(\d)', r'\1 \2', texto)
    return re.sub(r'\s+', ' ', texto).strip()


# ── expansão opcional de siglas (desligada por padrão; decidida na validação) ─
_SIGLAS_COR = {r'\bPT\.?': 'PRETO', r'\bBR\.?': 'BRANCO', r'\bAZ\.?': 'AZUL',
               r'\bVD\.?': 'VERDE', r'\bVM\.?': 'VERMELHO', r'\bAM\.?': 'AMARELO'}
_GATILHO_COR = re.compile(r'\b(CABO|FIO)\b', re.IGNORECASE)
_GATILHO_SERIE = re.compile(r'\b(ESGOTO|ESG)\b', re.IGNORECASE)


def expandir_siglas(texto):
    """Expande siglas só quando o próprio texto contém a palavra que as torna
    inequívocas (sem usar categoria)."""
    if _GATILHO_COR.search(texto):
        for padrao, cheio in _SIGLAS_COR.items():
            texto = re.sub(padrao + r'(?=\s|$)', lambda m: f'{m.group(0)} {cheio}',
                           texto, flags=re.IGNORECASE)
    if _GATILHO_SERIE.search(texto):
        texto = re.sub(r'\bSN\b(?!\s*SÉRIE)', lambda m: f'{m.group(0)} SÉRIE NORMAL',
                       texto, flags=re.IGNORECASE)
    return texto


# ── construção dos textos ───────────────────────────────────────────────────
def _campo(valor):
    return normalizar_numeros_unidades(limpar_ruido((valor or '').strip()))


def campos_limpos(item):
    """Devolve (nome, descricao, unidade) limpos. A descrição é descartada
    quando repete o nome (campo redundante duplicaria os termos)."""
    nome = _campo(item.get('nome_item'))
    desc = _campo(item.get('descricao_item'))
    uni = _campo(item.get('unidade'))
    if desc and desc.casefold() == nome.casefold():
        desc = ''
    return nome, desc, uni


def texto_a(item, siglas=False):
    """Variante A: concatenação simples (BM25 e variante A neural)."""
    t = ' '.join(c for c in campos_limpos(item) if c)
    return expandir_siglas(t) if siglas else t


def texto_b(item, siglas=False):
    """Variante B: serialização [COL] rótulo [VAL] valor (DITTO, §2.3)."""
    nome, desc, uni = campos_limpos(item)
    partes = []
    for rotulo, valor in (('nome', nome), ('descricao', desc), ('unidade', uni)):
        if valor:
            v = expandir_siglas(valor) if siglas else valor
            partes.append(f'[COL] {rotulo} [VAL] {v}')
    return ' '.join(partes)


def chave_dedup(item):
    """Itens com a mesma chave (na mesma empresa) são o mesmo texto para o modelo."""
    return unicodedata.normalize('NFC', texto_a(item)).casefold()
