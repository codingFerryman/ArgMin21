from typing import Dict

from transformers import AutoModel, AutoConfig


def model_map(model_config: Dict):
    if model_config['type'] == 'transformer':
        transformer_config = AutoConfig.from_pretrained(model_config['name_or_path'])
        transformer_config.update(model_config)
        return AutoModel.from_config(transformer_config)
    else:
        raise NotImplementedError(f"{model_config['type']} model doesn't be supported")

#
# def tokenizer_map(tokenizer_config: Dict):
#     if tokenizer_config['type'] == 'transformer':
#         if tokenizer_config['name_or_path'] == 'bert-':
#             return BertTokenizer.from_pretrained(tokenizer_config['name_or_path'])
#         elif tokenizer_config['name_or_path'] == 'roberta-':
#             return RobertaTokenizer.from_pretrained(tokenizer_config['name_or_path'])
#         elif tokenizer_config['name_or_path'] == 'albert-':
#             return AlbertTokenizer.from_pretrained(tokenizer_config['name_or_path'])
#         else:
#             return AutoTokenizer.from_pretrained(tokenizer_config['name_or_path'], use_fast=False)
#     else:
#         raise NotImplementedError(f"{tokenizer_config['type']} model doesn't be supported")
