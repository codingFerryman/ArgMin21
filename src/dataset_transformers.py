import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import get_data_path
from data.code.track_1_kp_matching import load_kpm_data


def generate_labeled_sentence_pair_df(subset="train"):
    assert subset in ["train", "dev", "test"]
    if subset == "test":
        gold_data_dir = Path(get_data_path(), 'test_data')
    else:
        gold_data_dir = Path(get_data_path(), 'kpm_data')
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset=subset)
    arg_df = arg_df[['arg_id', 'argument']]
    kp_df = kp_df[['key_point_id', 'key_point']]
    labels_df = pd.merge(labels_df, arg_df, on='arg_id')
    labels_df = pd.merge(labels_df, kp_df, on='key_point_id')
    return labels_df


class TransformersSentencePairDataset(Dataset):
    def __init__(self, model_name_or_path, max_len, subset="train"):
        self.data = generate_labeled_sentence_pair_df(subset=subset)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arg_sentence = str(self.data.loc[idx, 'argument']).lower()
        key_sentence = str(self.data.loc[idx, 'key_point']).lower()

        encoded_pair = self.tokenizer(
            arg_sentence, key_sentence,
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_len,
            return_tensors='pt'
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        label = self.data.loc[idx, 'label']

        return {'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label}
