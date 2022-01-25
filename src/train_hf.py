import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda
from sklearn.metrics import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier_hf import training
from evaluation import predict
from utils import get_project_path, get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = get_logger("main", "debug")

config_dir = Path(get_project_path(), 'config')

config_or_modelpath_list = [
    # "/home/he/Workspace/ArgMin21/models/roberta-base BM+TH F1_20220107-125459",
    # "/home/he/Workspace/ArgMin21/models/albert-base_20211215-205308"
    "/home/he/Workspace/ArgMin21/models/roberta-base_20211215-145739"
]

report_path = Path(Path(__file__).parent.resolve(), "report_new.csv")
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
    golden_dev, pred_dev, match_prob_dev = prediction_dev_df.golden_label, prediction_dev_df.prediction, prediction_dev_df.match_prob
    tn_dev, fp_dev, fn_dev, tp_dev = confusion_matrix(golden_dev, pred_dev).ravel()

    prediction_test_df, experiment_config = predict(model, tokenizer, experiment_config, "test")
    golden_test, pred_test, match_prob_test = prediction_test_df.golden_label, prediction_test_df.prediction, prediction_test_df.match_prob
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(golden_test, pred_test).ravel()

    neg_pred = [pred_test[i] for i, v in enumerate(golden_test) if v == 0]
    neg_pos_prob = [match_prob_test[i] for i, v in enumerate(golden_test) if v == 0]
    neg_true = np.zeros(len(neg_pred), dtype=int)
    pos_pred = [pred_test[i] for i, v in enumerate(golden_test) if v == 1]
    pos_pos_prob = [match_prob_test[i] for i, v in enumerate(golden_test) if v == 1]
    pos_true = np.ones(len(pos_pred), dtype=int)

    name = experiment_config.get("name", "default")
    model_report = {
        f"{name}": {
            "epoch_stop": state['epoch'],
            "mode": experiment_config['eval_config'].get('mode', 'plain') + str(experiment_config.get('threshold', '')),
            # "acc_dev": accuracy_score(golden_dev, pred_dev),
            # "bal_acc_dev": balanced_accuracy_score(golden_dev, pred_dev),
            # "precis_dev": precision_score(golden_dev, pred_dev),
            # "recall_dev": recall_score(golden_dev, pred_dev),
            # "f1_dev": f1_score(golden_dev, pred_dev),
            # "tnr_dev": tn_dev / (tn_dev + fp_dev),
            # "tpr_dev": tp_dev / (tp_dev + fn_dev),
            # "auc_dev": roc_auc_score(golden_dev, pred_dev),
            "acc_neg": accuracy_score(neg_true, neg_pred),
            "acc_pos": accuracy_score(pos_true, pos_pred),
            "acc": accuracy_score(golden_test, pred_test),
            # "prec_pos": precision_score(pos_true, pos_pred),
            "prec": precision_score(golden_test, pred_test),
            # "recall_pos": recall_score(pos_true, pos_pred),
            "recall": recall_score(golden_test, pred_test),
            "f1_pos": f1_score(pos_true, pos_pred),
            "f1": f1_score(golden_test, pred_test),
            "auc": roc_auc_score(golden_test, pred_test),
            # "config_path": str(config_path),
            "model_path": str(model_path)
        }
    }

    report_dict.update(model_report)
    report_df = pd.DataFrame.from_dict(report_dict, orient='index').fillna(-1)
    report_df.index.name = 'name'
    report_df.to_csv(report_path, float_format='%.5f')
