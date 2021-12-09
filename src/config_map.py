from typing import Dict, Callable
from transformers import AutoModel, AutoConfig, AutoTokenizer


def model_map(model_config: Dict):
    if model_config['type'] == 'transformer':
        transformer_config = AutoConfig.from_pretrained(model_config['name_or_path'])
        transformer_config.update(model_config.get("args", {}))
        return AutoModel.from_config(transformer_config)
    else:
        raise NotImplementedError(f"{model_config['type']} model doesn't be supported")


def tokenizer_map(tokenizer_config: Dict):
    if tokenizer_config['type'] == 'transformer':
        return AutoTokenizer.from_pretrained(tokenizer_config['name_or_path'], use_fast=False)
    else:
        raise NotImplementedError(f"{tokenizer_config['type']} model doesn't be supported")