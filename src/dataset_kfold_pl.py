import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer

from src.dataset_pl import KPMDataset
from src.utils import generate_labeled_sentence_pair_df


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


@dataclass
class KPMFoldDataModule(BaseKFoldDataModule):

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

        self.fit_dataset = None
        self.test_dataset = None

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)

    @staticmethod
    def string_preprocessing(text: str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        url = re.compile(r'https?://\S+|www\.\S+')
        text = url.sub(r'', text)
        html = re.compile(r'<.*?>')
        text = html.sub(r'', text)

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text

    def convert_to_features(self, example_dataframe):
        tasks = self.task_name.split(',')
        if 'sentence_pair' in tasks:
            features = self.tokenizer.batch_encode_plus(
                list(zip(example_dataframe['argument'].apply(lambda x: self.string_preprocessing(x)).tolist(),
                         example_dataframe['key_point'].apply(lambda x: self.string_preprocessing(x)).tolist())),
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

    def setup(self, stage: Optional[str] = None) -> None:
        # if stage == 'fit':
        train_pair_df = generate_labeled_sentence_pair_df('train')
        # self.train_dataset = KPMDataset(self.convert_to_features(train_pair_df))
        val_pair_df = generate_labeled_sentence_pair_df('dev')
        # self.val_dataset = KPMDataset(self.convert_to_features(val_pair_df))
        self.fit_pair_df = pd.concat([train_pair_df, val_pair_df])
        self.fit_dataset = KPMDataset(self.convert_to_features(self.fit_pair_df))
        # if stage == 'test':
        test_pair_df = generate_labeled_sentence_pair_df('test')
        self.test_dataset = KPMDataset(self.convert_to_features(test_pair_df))

    def setup_folds(self, num_folds: int = 5) -> None:
        self.num_folds = num_folds
        uni_args = list(set(self.fit_pair_df.arg_id))
        num_args = len(uni_args)
        self.args_id_dict = {i: v for i, v in enumerate(uni_args)}
        self.splits = [split for split in KFold(num_folds).split(range(num_args))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_arg_indices, val_arg_indices = self.splits[fold_index]
        train_args = [self.args_id_dict[i] for i in train_arg_indices]
        train_indices = self.fit_pair_df.index[self.fit_pair_df.arg_id.isin(train_args)]
        val_args = [self.args_id_dict[i] for i in val_arg_indices]
        val_indices = self.fit_pair_df.index[self.fit_pair_df.arg_id.isin(val_args)]
        self.train_fold = Subset(self.fit_dataset, train_indices)
        self.val_fold = Subset(self.fit_dataset, val_indices)

    def fit_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.fit_dataset,
            batch_size=self.train_batch_size,
            # num_workers=12,
            shuffle=True
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_fold,
            batch_size=self.train_batch_size,
            num_workers=12,
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_fold,
            batch_size=self.eval_batch_size,
            num_workers=12,
            # shuffle=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=12,
        )
