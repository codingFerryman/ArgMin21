from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from transformers import AutoTokenizer

from utils import get_data_path
from data.code.track_1_kp_matching import load_kpm_data


def generate_labeled_sentence_pair_df():
    gold_data_dir = Path(get_data_path(), 'kpm_data')
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="train")
    arg_df = arg_df[['arg_id', 'argument']]
    kp_df = kp_df[['key_point_id', 'key_point']]
    labels_df = pd.merge(labels_df, arg_df, on='arg_id')
    labels_df = pd.merge(labels_df, kp_df, on='key_point_id')
    return labels_df


class TransformersSentencePairDataset(Dataset):
    def __init__(self, model_name_or_path, max_len):
        self.data = generate_labeled_sentence_pair_df()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arg_sentence = str(self.data.loc[idx, 'argument'])
        key_sentence = str(self.data.loc[idx, 'key_point'])

        encoded_pair = self.tokenizer(
            arg_sentence, key_sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        label = self.data.loc[idx, 'label']

        return token_ids, attn_masks, token_type_ids, label