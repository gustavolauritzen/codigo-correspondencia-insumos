"""Garantia de que Bi-Encoder e Cross-Encoder usam o BERTimbau LARGE, em versões fixadas.

Problema conhecido: o identificador "...-base-..." do Bi-Encoder pode ser redirecionado pelo Hugging Face para outro
modelo (Large no lugar de Base), e os dois modelos precisam ser o mesmo BERT Large. Aqui o identificador, a revisão e a arquitetura são conferidos antes de qualquer
treino ou avaliação; qualquer divergência interrompe a execução.
"""
import config as C

ESPERADO = {'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'vocab_size': 29794}


def conferir_arquitetura(cfg):
    """`cfg`: dicionário (ou objeto) com a configuração do modelo. Levanta ValueError se não for o BERTimbau Large."""
    for k, v in ESPERADO.items():
        achado = cfg.get(k) if isinstance(cfg, dict) else getattr(cfg, k, None)
        if achado != v:
            raise ValueError(f'modelo fora do BERTimbau Large: {k} = {achado} (esperado {v})')
    return True


def conferir_identificadores():
    for nome in (C.MODELO_BI_ENCODER, C.MODELO_CROSS_ENCODER):
        if 'large' not in nome.lower() or 'base' in nome.lower():
            raise ValueError(f'identificador não é Large: {nome}')
    for rev in (C.MODELO_BI_ENCODER_REV, C.MODELO_CROSS_ENCODER_REV):
        if len(rev) != 40 or any(c not in '0123456789abcdef' for c in rev):
            raise ValueError(f'revisão não fixada: {rev}')
    return True


def carregar_config_verificada(nome, revisao=None):
    """Baixa só o config.json (transformers) e confere. Para pastas locais, `revisao` é ignorada."""
    import os
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(nome, revision=None if os.path.isdir(nome) else revisao)
    conferir_arquitetura(cfg)
    return cfg
