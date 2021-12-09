import random
from pathlib import Path
import numpy as np
import torch
import logging
import coloredlogs
import psutil
import humanize
import GPUtil
import os
from ignite.utils import manual_seed
import ignite.distributed as idist


def set_seed(seed: int = 2233):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    Args:
        seed (:obj:`int`): The seed to set.
    """
    manual_seed(seed)
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


def get_device():
    return idist.device()


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
