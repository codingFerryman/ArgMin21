import json

import pandas as pd
import torch.cuda
from pathlib import Path
from sklearn.metrics import *

from evaluation import predict
from transformers_pipeline import training
from utils import print_mem, get_project_path, get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger("main", "debug")

config_dir = Path(get_project_path(), 'config')

config_or_modelpath_list = [
    "albert-base.json"
]

report_path = Path(Path(__file__).parent.resolve(), "report.csv")
if Path(report_path).is_file():
    _tmp_report_df = pd.read_csv(report_path, index_col='name')
    report_dict = _tmp_report_df.to_dict('index')
else:
    report_dict = {}

for config_or_modelpath in config_or_modelpath_list:
    torch.cuda.empty_cache()
    logger.info(f"Training: {config_or_modelpath}")

    if config_or_modelpath[-5:] == '.json':
        config_path = Path(config_dir, config_or_modelpath)

        with open(config_path, 'r') as fc:
            experiment_config = json.load(fc)

        trainer, model_path = training(config_path)
    else:
        model_path = config_or_modelpath
        with open(Path(model_path, 'training.json'), 'r') as fc:
            experiment_config = json.load(fc)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(Path(model_path, 'state.json'), 'r') as fs:
        state = json.load(fs)

    prediction_dev_df, experiment_config = predict(model, tokenizer, experiment_config, "dev")
    golden_dev, pred_dev = prediction_dev_df.golden_label, prediction_dev_df.prediction
    tn_dev, fp_dev, fn_dev, tp_dev = confusion_matrix(golden_dev, pred_dev).ravel()

    prediction_test_df, experiment_config = predict(model, tokenizer, experiment_config, "test")
    golden_test, pred_test = prediction_test_df.golden_label, prediction_test_df.prediction
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(golden_test, pred_test).ravel()

    name = experiment_config.get("name", "default")
    model_report = {
        f"{name}": {
            "epoch_stop": state['epoch'],
            "mode": experiment_config['eval_config'].get('mode', 'plain') + str(experiment_config.get('threshold', '')),
            "acc_dev": accuracy_score(golden_dev, pred_dev),
            "bal_acc_dev": balanced_accuracy_score(golden_dev, pred_dev),
            "f1_dev": f1_score(golden_dev, pred_dev),
            "tnr_dev": tn_dev / (tn_dev + fp_dev),
            "tpr_dev": tp_dev / (tp_dev + fn_dev),
            "auc_dev": roc_auc_score(golden_dev, pred_dev),
            "acc_test": accuracy_score(golden_test, pred_test),
            "bal_acc_test": balanced_accuracy_score(golden_test, pred_test),
            "f1_test": f1_score(golden_test, pred_test),
            "tnr_test": tn_test / (tn_test + fp_test),
            "tpr_test": tp_test / (tp_test + fn_test),
            "auc_test": roc_auc_score(golden_test, pred_test),
            # "config_path": str(config_path),
            "model_path": str(model_path)
        }
    }

    report_dict.update(model_report)
    report_df = pd.DataFrame.from_dict(report_dict, orient='index')
    report_df.index.name = 'name'
    report_df.to_csv(report_path, float_format='%.5f')
