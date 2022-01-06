from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoTokenizer, BatchEncoding, BertTokenizer

from src.utils import get_logger, load_kpm_data, generate_labeled_sentence_pair_df

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class KPMDataset(Dataset):
    def __init__(self, batched_encodings: BatchEncoding):
        self.batched_encodings = batched_encodings

    def __len__(self):
        return len(self.batched_encodings.data['input_ids'])

    def __getitem__(self, idx):
        result = dict()
        for _k in self.batched_encodings.data.keys():
            result[_k] = self.batched_encodings.data[_k][idx]
        return result


class KPMDataModule(LightningDataModule):

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "sentence_pair",
            num_labels: int = 1,
            max_seq_length: int = 96,
            train_batch_size: int = 32,
            eval_batch_size: int = 16,
            **kwargs,
    ):
        super().__init__()

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            train_pair_df = generate_labeled_sentence_pair_df('train')
            self.train_dataset = KPMDataset(self.convert_to_features(train_pair_df))
            val_pair_df = generate_labeled_sentence_pair_df('dev')
            self.val_dataset = KPMDataset(self.convert_to_features(val_pair_df))
        if stage == 'test':
            test_pair_df = generate_labeled_sentence_pair_df('test')
            self.test_dataset = KPMDataset(self.convert_to_features(test_pair_df))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=12,
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=12,
            # shuffle=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def convert_to_features(self, example_dataframe):
        tasks = self.task_name.split(',')
        if 'sentence_pair' in tasks:
            features = self.tokenizer.batch_encode_plus(
                list(zip(example_dataframe['argument'].str.lower().tolist(),
                         example_dataframe['key_point'].str.lower().tolist())),
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
        else:
            raise NotImplementedError
        features["labels"] = example_dataframe["label"].tolist()
        return features
