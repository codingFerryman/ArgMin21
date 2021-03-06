import logging
import os
import random
import re
from pathlib import Path

import GPUtil
import coloredlogs
import humanize
import pandas as pd
import psutil
import torch
from pytorch_lightning import seed_everything


def load_kpm_data(gold_data_dir=None, subset='train', submitted_kp_file=None):
    if gold_data_dir is None:
        if subset != 'test':
            gold_data_dir = Path(get_data_path(), 'kpm_data')
        else:
            gold_data_dir = Path(get_data_path(), 'test_data')

    get_logger(f"Loading {subset} data:")
    arguments_file = Path(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = Path(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file = submitted_kp_file

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)

    # Please be aware that this file is not available during workshop
    labels_file = Path(gold_data_dir, f"labels_{subset}.csv")
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df


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


def generate_combined_df(subset="train", ratio=1.):
    assert subset in ["train", "dev", "test", "test_eval"]
    if subset == "test":
        gold_data_dir = Path(get_data_path(), 'test_data')
        return _generate_combined_df_test(gold_data_dir)
    else:
        if subset == 'test_eval':
            subset = "test"
    if subset == 'test':
        gold_data_dir = Path(get_data_path(), 'test_data')
    else:
        gold_data_dir = Path(get_data_path(), 'kpm_data')
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset=subset)
    # arg_df = arg_df[['arg_id', 'argument']]
    kp_df = kp_df[['key_point_id', 'key_point']]
    labels_df = pd.merge(labels_df, arg_df, on='arg_id')
    labels_df = pd.merge(labels_df, kp_df, on='key_point_id')

    if ratio < 1.:
        all_args = list(set(labels_df.arg_id))
        num_args = len(all_args)
        select_num_args = int(num_args * ratio)
        select_args = random.sample(all_args, select_num_args)
        labels_df = labels_df[labels_df.arg_id.isin(select_args)].reset_index()

    return labels_df[['arg_id', 'argument', 'key_point_id', 'key_point', 'topic', 'stance', 'label']]


def _generate_combined_df_test(data_dir):
    arg_df, kp_df, label_df = load_kpm_data(data_dir, subset='test')
    arg_df["topic_id"] = arg_df["arg_id"].map(extract_topic)
    kp_df["topic_id"] = kp_df["key_point_id"].map(extract_topic)
    merged_df = pd.merge(arg_df, kp_df, how="left", on=["topic", "stance", "topic_id"])
    merged_df = merged_df.join(label_df.set_index(['arg_id', 'key_point_id']), on=['arg_id', 'key_point_id'])
    merged_df['labels'] = None
    return merged_df[['arg_id', 'argument', 'key_point_id', 'key_point', 'topic', 'stance', 'label']]


def extract_topic(text: str) -> int:
    topic_id = text.split("_")[1]
    return int(topic_id)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int = 42):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    Args:
        seed (:obj:`int`): The seed to set.
    """
    seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_project_path() -> Path:
    """The function for getting the root directory of the project"""
    try:
        import git
        return git.Repo(Path(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    except NameError:
        return Path(__file__).parent.parent


def get_data_path() -> Path:
    return Path(get_project_path(), 'data')


def get_logger(name: str, level='debug'):
    fmt = '[%(asctime)s] - %(name)s - {line:%(lineno)d} %(levelname)s - %(message)s'
    logger = logging.getLogger(name=name)
    if level.lower() == 'debug':
        logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=fmt, level='DEBUG', logger=logger)
    elif level.lower() == 'info':
        logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=fmt, level='INFO', logger=logger)
    elif level.lower() in ['warn', 'warning']:
        logger.setLevel(logging.WARNING)
        coloredlogs.install(fmt=fmt, level='WARNING', logger=logger)
    return logger


def print_mem():
    gpus = GPUtil.getGPUs()
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().inactive),
          " | Proc size: " + humanize.naturalsize(process.memory_info().rss)
          )
    for gpu in gpus:
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(
            gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal)
        )
