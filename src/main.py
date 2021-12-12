import json

import pandas as pd
import torch.cuda
from pathlib import Path
from sklearn.metrics import *

from evaluation import predict
from transformers_pipeline import training
from utils import print_mem, get_project_path, get_logger

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger("main", "debug")

config_dir = Path(get_project_path(), 'config')

config_file_list = [
    # "albert-base.json",
    "roberta-base.json"
]

report_path = Path(Path(__file__).parent.resolve(), "report.csv")
if Path(report_path).is_file():
    _tmp_report_df = pd.read_csv(report_path, index_col='name')
    report_dict = _tmp_report_df.to_dict('index')
else:
    report_dict = {}

report_args = [
    "name",
    "epoch_stop",
    "threshold",
    "mode",
    "acc_dev",
    "bal_acc_dev",
    "pre_dev",
    "rec_dev",
    "f1_dev",
    "acc_test",
    "bal_acc_test",
    "pre_test",
    "rec_test",
    "f1_test",
    "config_path",
    "model_path"
]

for config in config_file_list:
    torch.cuda.empty_cache()
    logger.info(f"Training config: {config}")
    config_path = Path(config_dir, config)

    with open(config_path, 'r') as fc:
        experiment_config = json.load(fc)

    trainer, model_path = training(config_path)
    model = trainer.model

    prediction_test_df = predict(model, experiment_config, "test")
    golden_test, pred_test = prediction_test_df.golden_label, prediction_test_df.prediction

    prediction_dev_df = predict(model, experiment_config, "dev")
    golden_dev, pred_dev = prediction_dev_df.golden_label, prediction_dev_df.prediction

    name = experiment_config.get("name", "default")
    model_report = {
        f"{name}": {
            "epoch_stop": experiment_config["trainer_config"].get("num_train_epochs", 5),
            "mode": experiment_config['eval_config'].get('mode', 'plain'),
            "acc_dev": accuracy_score(golden_dev, pred_dev),
            "bal_acc_dev": balanced_accuracy_score(golden_dev, pred_dev),
            "pre_dev": precision_score(golden_dev, pred_dev),
            "rec_dev": recall_score(golden_dev, pred_dev),
            "f1_dev": f1_score(golden_dev, pred_dev),
            "acc_test":accuracy_score(golden_test, pred_test),
            "bal_acc_test": balanced_accuracy_score(golden_test, pred_test),
            "pre_test": precision_score(golden_test, pred_test),
            "rec_test": recall_score(golden_test, pred_test),
            "f1_test": f1_score(golden_test, pred_test),
            "config_path": str(config_path),
            "model_path": str(model_path)
        }
    }

    report_dict.update(model_report)
    report_df = pd.DataFrame.from_dict(report_dict, orient='index')
    report_df.index.name = 'name'
    report_df.to_csv(report_path)




