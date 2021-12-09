import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

from utils import get_data_path
from config_map import tokenizer_map


def load_kpm_data(gold_data_dir, subset, submitted_kp_file=None):
    print(f"\nֿ** loading {subset} data:")
    arguments_file = Path(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = Path(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file = submitted_kp_file
    labels_file = Path(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    for desc, group in arguments_df.groupby(["stance", "topic"]):
        stance = desc[0]
        topic = desc[1]
        key_points = key_points_df[(key_points_df["stance"] == stance) & (key_points_df["topic"] == topic)]
        print(f"\t{desc}: loaded {len(group)} arguments and {len(key_points)} key points")
    print("\n")
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
    def __init__(self, tokenizer_config, subset="train"):
        self.data = generate_labeled_sentence_pair_df(subset=subset)
        self.tokenizer = tokenizer_map(tokenizer_config)
        self.tokenizer_config = tokenizer_config.get('args', {})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arg_sentence = str(self.data.loc[idx, 'argument']).lower()
        key_sentence = str(self.data.loc[idx, 'key_point']).lower()

        encoded_pair = self.tokenizer(
            arg_sentence, key_sentence,
            **self.tokenizer_config
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        label = self.data.loc[idx, 'label']

        return {'input_ids': token_ids,
                'attention_mask': attn_masks,
                'token_type_ids': token_type_ids,
                'labels': label}
