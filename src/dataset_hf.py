import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import Dataset

from utils import get_data_path, get_logger, generate_labeled_sentence_pair_df
from config_map import tokenizer_map

LOG_LEVEL = "INFO"
logger = get_logger("dataset", level=LOG_LEVEL)


class TransformersSentencePairDataset(Dataset):
    def __init__(self, tokenizer_config=None, subset="train", pretrained_tokenizer=None):
        self.subset = subset
        self.data = generate_labeled_sentence_pair_df(subset=subset)
        if pretrained_tokenizer is not None:
            self.tokenizer = pretrained_tokenizer
        elif tokenizer_config is not None:
            self.tokenizer = tokenizer_map(tokenizer_config)
        self.tokenizer_config = tokenizer_config.get('args', {})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arg_sentence = str(self.data.loc[idx, 'argument']).lower()
        key_sentence = str(self.data.loc[idx, 'key_point']).lower()
        arg_id = self.data.loc[idx, 'arg_id']
        key_point_id = self.data.loc[idx, 'key_point_id']

        encoded_pair = self.tokenizer.encode_plus(
            arg_sentence, key_sentence,
            return_token_type_ids=True,
            return_attention_mask=True,
            **self.tokenizer_config
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        label = self.data.loc[idx, 'label']
        if self.subset == 'train':
            return {
                'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label,
            }
        else:
            return {
                'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label,
                'arg_id': arg_id,  # List
                'key_point_id': key_point_id  # List
            }
