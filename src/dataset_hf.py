from typing import Optional, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import get_logger, generate_combined_df, string_preprocessing

LOG_LEVEL = "INFO"
logger = get_logger("dataset", level=LOG_LEVEL)


class KPADataset(Dataset):
    def __init__(
            self,
            tokenizer_name,
            tokenizer_config=None,
            data: Optional[pd.DataFrame] = None,
            subset: Optional[str] = "train",
            pretrained_tokenizer=None,
            load_ratio=1.,
            add_info: Optional[List] = None
    ):
        assert bool(data) + bool(subset), "None of data or subset is provided :("

        if bool(data) and bool(subset):
            self.subset = None
            self.data = data
        else:
            self.subset = subset
            self.data = generate_combined_df(self.subset, ratio=load_ratio)

        if pretrained_tokenizer is not None:
            self.tokenizer = pretrained_tokenizer
        elif tokenizer_config is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer_config = tokenizer_config

        self.add_info = add_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arg_sentence = self.text_preprocessing(str(self.data.loc[idx, 'argument']))
        key_sentence = self.text_preprocessing(str(self.data.loc[idx, 'key_point']))
        encoded_pair = self.tokenizer.encode_plus(
            arg_sentence, key_sentence,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length',
            truncation="longest_first",
            **self.tokenizer_config
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        incl_stance = False

        if self.add_info is not None:
            if 'stance' in self.add_info:
                assert 'topic' in self.add_info, 'stance info has to be combined with topic'
                incl_stance = True
                stance = self.data.loc[idx, 'stance']
                if stance == 1:
                    stance = ' is positive'
                elif stance == -1:
                    stance = ' is negative'
                else:
                    raise ValueError
                self.add_info.remove('stance')

            for info in self.add_info:
                preprocessed_text = self.text_preprocessing(str(self.data.loc[idx, info]))
                if info == 'topic' and incl_stance:
                    preprocessed_text = preprocessed_text + stance
                tokenizer_config = self.tokenizer_config
                if info == 'topic':
                    tokenizer_config.update({"max_length": 10})
                encoded = self.tokenizer.encode_plus(
                    preprocessed_text,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    padding='max_length',
                    truncation=True,
                    **tokenizer_config
                )
                token_ids = torch.cat([token_ids, encoded['input_ids'].squeeze(0)])
                attn_masks = torch.cat([attn_masks, encoded['attention_mask'].squeeze(0)])
                token_type_ids = torch.cat([token_type_ids, encoded['token_type_ids'].squeeze(0)])

        arg_id = self.data.loc[idx, 'arg_id']
        key_point_id = self.data.loc[idx, 'key_point_id']

        label = self.data.loc[idx, 'label']
        if self.subset == 'test':
            return {
                'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label,
                'arg_id': arg_id,  # List
                'key_point_id': key_point_id  # List
            }
        else:
            return {
                'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label
            }

    @staticmethod
    def text_preprocessing(text: str):
        return string_preprocessing(text)
