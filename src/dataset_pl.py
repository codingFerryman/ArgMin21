import os
import re
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BatchEncoding

from src.utils import generate_labeled_sentence_pair_df

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
