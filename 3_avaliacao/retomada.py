"""Salvar progresso e retomar de onde parou. Sem torch: testável localmente.

Contexto: sessões do Colab (ou de qualquer ambiente) caem por limite de
tempo/cota de GPU e um disco montado (por exemplo o Drive, via FUSE) pode falhar em silêncio. Regras:
  1. Resultados por consulta são gravados em CSV em modo append, com flush + fsync, e a retomada pula as consultas já
     gravadas; as métricas só são calculadas depois que TODAS as consultas existem, relidas do disco.
  2. Estado de treino (modelo, otimizador, agenda, histórico) é gravado ao fim de cada época em pasta temporária e
     renomeado (nunca meio escrito); depois é copiado para a pasta persistente e conferido por tamanho e MD5.
  3. Uma execução com `concluido.json` nunca é refeita nem sobrescrita; sem ele, a execução continua do último estado.
  4. Toda escrita de arquivo pequeno é atômica (arquivo temporário + os.replace).
"""
import csv
import hashlib
import json
import os
import shutil
import time

import metricas as M
import tempo


# ── arquivos pequenos ────────────────────────────────────────────────────────
def escrita_atomica(caminho, texto):
    os.makedirs(os.path.dirname(caminho) or '.', exist_ok=True)
    tmp = f'{caminho}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(texto)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, caminho)


def gravar_json(caminho, obj):
    escrita_atomica(caminho, json.dumps(obj, ensure_ascii=False, indent=1))


def ler_json(caminho, padrao=None):
    if not os.path.exists(caminho):
        return padrao
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


def concluido(pasta):
    return os.path.exists(os.path.join(pasta, 'concluido.json'))


# ── pastas grandes (checkpoints) ─────────────────────────────────────────────
def md5_arquivo(caminho, bloco=1 << 22):
    h = hashlib.md5()
    with open(caminho, 'rb') as f:
        for parte in iter(lambda: f.read(bloco), b''):
            h.update(parte)
    return h.hexdigest()


def _arquivos(pasta):
    for raiz, _, nomes in os.walk(pasta):
        for n in sorted(nomes):
            caminho = os.path.join(raiz, n)
            yield os.path.relpath(caminho, pasta), caminho


MD5_ATE_BYTES = 200 * 1024 * 1024      # arquivos maiores que isso (pesos do modelo) são conferidos só por tamanho


def copiar_arquivo_verificado(origem, destino):
    """Copia um arquivo pequeno (por exemplo o CSV de progresso) de forma atômica e confere tamanho e MD5."""
    os.makedirs(os.path.dirname(destino) or '.', exist_ok=True)
    tmp = f'{destino}.tmp'
    shutil.copyfile(origem, tmp)
    if os.path.getsize(tmp) != os.path.getsize(origem) or md5_arquivo(tmp) != md5_arquivo(origem):
        os.remove(tmp)
        raise IOError(f'cópia inconsistente de {origem}')
    os.replace(tmp, destino)


def copiar_verificado(origem, destino):
    """Copia a pasta `origem` para `destino` de forma atômica: copia para `destino.tmp`, confere tamanho e MD5 de cada
    arquivo, renomeia. Se `destino` existia, ele é substituído só DEPOIS da conferência (fica `destino.antigo` por um
    instante). Levanta IOError se algo não bater (nada é dado como salvo)."""
    tmp = f'{destino}.tmp'
    antigo = f'{destino}.antigo'
    for p in (tmp, antigo):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(os.path.dirname(destino) or '.', exist_ok=True)
    shutil.copytree(origem, tmp)
    for rel, caminho in _arquivos(origem):
        c2 = os.path.join(tmp, rel)
        pequeno = os.path.getsize(caminho) <= MD5_ATE_BYTES
        if not os.path.exists(c2) or os.path.getsize(c2) != os.path.getsize(caminho) or (pequeno and md5_arquivo(c2) != md5_arquivo(caminho)):
            shutil.rmtree(tmp, ignore_errors=True)
            raise IOError(f'cópia inconsistente em {rel}: nada foi substituído em {destino}')
    if os.path.exists(destino):
        os.rename(destino, antigo)
    os.rename(tmp, destino)
    shutil.rmtree(antigo, ignore_errors=True)


def restaurar_se_faltar(persistente, local):
    """Se `local` não existe (sessão nova) e `persistente` existe, traz o conteúdo para o disco local."""
    if os.path.exists(local) or not os.path.exists(persistente):
        return False
    copiar_verificado(persistente, local)
    return True


# ── resultados por consulta ──────────────────────────────────────────────────
CAMPOS = ['query_id', 'n_rel', 'ap', 'rr', 'rp', 'h1', 'h5', 'h10', 'ms', 'gpu']


class RegistroConsultas:
    """CSV append-only com o resultado de cada consulta (métricas daquela consulta e o tempo em ms)."""

    def __init__(self, caminho):
        self.caminho = caminho
        os.makedirs(os.path.dirname(caminho) or '.', exist_ok=True)

    def feitas(self):
        if not os.path.exists(self.caminho):
            return set()
        try:
            with open(self.caminho, encoding='utf-8', newline='') as f:
                return {r['query_id'] for r in csv.DictReader(f) if r.get('ms') not in (None, '')}
        except Exception as e:                      # arquivo corrompido: não finge que há progresso
            raise IOError(f'{self.caminho} ilegível ({e}); confira o arquivo antes de continuar') from e

    def acrescentar(self, linha):
        novo = not os.path.exists(self.caminho)
        with open(self.caminho, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS, lineterminator='\n')
            if novo:
                w.writeheader()
            w.writerow(linha)
            f.flush()
            os.fsync(f.fileno())

    def linhas(self):
        if not os.path.exists(self.caminho):
            return []
        with open(self.caminho, encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))


def executar_consultas(registro, consultas, relevantes, funcao_ranquear, cronometro, gpu='cpu', rotulo='', persistir=None, cada=100):
    """Roda `funcao_ranquear(consulta) -> ranking` nas consultas que ainda não estão no registro, gravando cada uma
    na hora. Devolve (quantas rodou nesta sessão, quantas já estavam feitas)."""
    ja = registro.feitas()
    rodou = 0
    alvo = [q for q in consultas if q['id'] in relevantes and relevantes[q['id']]]
    for q in alvo:
        if q['id'] in ja:
            continue
        t0 = len(cronometro.consultas_ms)
        ranking = cronometro.consulta(funcao_ranquear, q)
        ms = cronometro.consultas_ms[-1] if len(cronometro.consultas_ms) > t0 else 0.0
        linha = M.por_consulta(ranking, relevantes[q['id']])
        linha.update({'query_id': q['id'], 'ms': f'{ms:.4f}', 'gpu': gpu})
        registro.acrescentar(linha)
        rodou += 1
        if rodou % cada == 0:
            if persistir:
                persistir()            # copia o CSV de progresso para a pasta persistente
            print(f'  {rotulo} {len(ja) + rodou}/{len(alvo)} consultas', flush=True)
    if persistir:
        persistir()
    return rodou, len(ja)


def agregar_registro(registro, consultas, relevantes, setup_ms=0.0):
    """Relê o CSV do disco e calcula as métricas; exige TODAS as consultas com relevante. Inclui Setup, Consulta e Total."""
    alvo = {q['id'] for q in consultas if q['id'] in relevantes and relevantes[q['id']]}
    linhas = [r for r in registro.linhas() if r['query_id'] in alvo]
    vistos = {r['query_id'] for r in linhas}
    if vistos != alvo:
        raise RuntimeError(f'incompleto: {len(vistos)}/{len(alvo)} consultas; rode de novo para completar')
    out = M.agregar(linhas)
    ms = [float(r['ms']) for r in linhas]
    out.update(tempo.resumir(setup_ms, ms))
    out['gpus'] = sorted({r['gpu'] for r in linhas})
    return out


class RegistroJsonl:
    """Arquivo JSON-lines append-only (um objeto por linha, com `chave`), com flush + fsync: guarda resultados parciais
    maiores que uma linha de métricas (por exemplo os scores do Cross-Encoder de cada consulta da validação)."""

    def __init__(self, caminho, chave='query_id'):
        self.caminho, self.chave = caminho, chave
        os.makedirs(os.path.dirname(caminho) or '.', exist_ok=True)

    def ler(self):
        if not os.path.exists(self.caminho):
            return {}
        saida = {}
        with open(self.caminho, encoding='utf-8') as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    obj = json.loads(linha)
                except json.JSONDecodeError:         # última linha cortada por queda da sessão: ignora, será refeita
                    continue
                saida[obj[self.chave]] = obj
        return saida

    def acrescentar(self, obj):
        with open(self.caminho, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
