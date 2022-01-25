import logging
import os
import random
from pathlib import Path

import GPUtil
import coloredlogs
import humanize
import pandas as pd
import psutil
from pytorch_lightning import seed_everything


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


def generate_labeled_sentence_pair_df(subset="train", ratio=1.):
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

    if ratio < 1.:
        all_args = list(set(labels_df.arg_id))
        num_args = len(all_args)
        select_num_args = int(num_args * ratio)
        select_args = random.sample(all_args, select_num_args)
        labels_df = labels_df[labels_df.arg_id.isin(select_args)].reset_index()

    return labels_df[['arg_id', 'argument', 'key_point_id', 'key_point', 'label']]


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
