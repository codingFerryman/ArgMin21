import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import Dataset

from utils import get_data_path, get_logger
from config_map import tokenizer_map

LOG_LEVEL = "INFO"
logger = get_logger("dataset", level=LOG_LEVEL)


def load_kpm_data(gold_data_dir, subset, submitted_kp_file=None, debug=False):
    get_logger(f"Loading {subset} data:")
    arguments_file = Path(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = Path(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file = submitted_kp_file
    labels_file = Path(gold_data_dir, f"labels_{subset}.csv")

    if (debug is True) and (subset != "test"):
        arguments_df = pd.read_csv(arguments_file).sample(frac=0.5)
        key_points_df = pd.read_csv(key_points_file).sample(frac=0.5)
        labels_file_df = pd.read_csv(labels_file).sample(frac=0.5)
    else:
        arguments_df = pd.read_csv(arguments_file)
        key_points_df = pd.read_csv(key_points_file)
        labels_file_df = pd.read_csv(labels_file)

    # for desc, group in arguments_df.groupby(["stance", "topic"]):
    #     stance = desc[0]
    #     topic = desc[1]
    #     key_points = key_points_df[(key_points_df["stance"] == stance) & (key_points_df["topic"] == topic)]
    #     print(f"\t{desc}: loaded {len(group)} arguments and {len(key_points)} key points")
    # print("\n")
    return arguments_df, key_points_df, labels_file_df


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
            return {'input_ids': token_ids,
                    'attention_mask': attn_masks,
                    'token_type_ids': token_type_ids,
                    'labels': label,
                    # 'arg_id': arg_id,   # List
                    # 'key_point_id': key_point_id    # List
                    }
        else:
            return {'input_ids': token_ids,
                    'attention_mask': attn_masks,
                    'token_type_ids': token_type_ids,
                    'labels': label,
                    'arg_id': arg_id,   # List
                    'key_point_id': key_point_id    # List
                    }
