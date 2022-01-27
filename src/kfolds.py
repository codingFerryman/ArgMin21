from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from src.utils import get_logger, generate_combined_df

logger = get_logger("dataset")


class KFolds:
    def __init__(self, num_folds: int = 7, folds_dir: Optional[Union[Path, str]] = None):
        self.num_folds = num_folds
        self.folds_dir = folds_dir
        if self.folds_dir is None:
            self.folds_dir = Path('.', 'folds')
            self.folds_dir.mkdir(parents=True, exist_ok=True)

        self.fit_df = None
        self.topic2id = None
        self.splits = None

    def load_data(self, ratio=1.):
        train_df = generate_combined_df(subset='train', ratio=ratio)
        dev_df = generate_combined_df(subset='dev', ratio=ratio)
        self.fit_df = pd.concat([train_df, dev_df]).drop_duplicates()

    def _assign_topic_ids(self):
        assert self.fit_df is not None
        _topics = set(self.fit_df.topic)
        self.topic2id = {t: i for i, t in enumerate(_topics)}

    def setup_folds(self):
        self._assign_topic_ids()
        self.fit_df['topic_id'] = self.fit_df['topic'].map(lambda x: self.topic2id[x])
        self.splits = [split for split in KFold(self.num_folds).split(range(len(self.topic2id)))]

    def get_fold(self, fold_id: int) -> Tuple[pd.DataFrame, ...]:
        assert self.splits is not None
        fold_split = self.splits[fold_id]
        fold_train = self.fit_df[self.fit_df.topic_id.isin(fold_split[0])]
        fold_dev = self.fit_df[self.fit_df.topic_id.isin(fold_split[1])]
        return fold_train, fold_dev

    def get_fold_dataset(self, fold_id: int, **dataset_config):
        fold_train, fold_dev = self.get_fold(fold_id)
        train_dataset = dataset_config.update({'data': fold_train})
        dev_dataset = dataset_config.update({'data': fold_dev})
        return train_dataset, dev_dataset


if __name__ == '__main__':
    _test_class = KFolds(7)
    _test_class.load_data()
    _test_class.setup_folds()
    _test_class.get_fold(1)
