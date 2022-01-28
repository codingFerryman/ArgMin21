import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch.cuda
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier import training
from evaluate import evaluate, generate_submission, calc_map
from predict import predict
from src.kfolds import KFolds
from utils import get_project_path, get_logger, set_seed

set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger("main", "debug")

config_dir = Path(get_project_path(), 'config')


def run_kfold(config_or_modelpath, cuda_device="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    torch.cuda.empty_cache()
    logger.info(f"Model: {config_or_modelpath}")

    config_path = Path(config_dir, config_or_modelpath)

    with open(config_path, 'r') as fc:
        experiment_config = json.load(fc)

    # ============================================
    # Train on folds
    # ============================================
    name = experiment_config.get("name", "default")
    num_folds = experiment_config.get('num_folds', 7)

    Folds = KFolds(num_folds)
    Folds.load_data(experiment_config['data_config'].get('load_ratio', 1.))
    Folds.setup_folds()

    final_report = {}
    folds_predictions = []

    for fold_i in range(num_folds):
        logger.info(f"Training on Fold {fold_i}")
        train_data, dev_data = Folds.get_fold(fold_i)
        fit_data_df = Folds.get_fit_df()

        _, model_path = training(
            config_path,
            # kfold_fit_dataset=fit_data_df,
            train_data=train_data,
            val_data=dev_data
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        with open(Path(model_path, 'state.json'), 'r') as fs:
            state = json.load(fs)

        # ============================================
        # Evaluate, predict, and report
        # ============================================

        # Report template
        logger.info(f"Fold {fold_i}: Preparing the report for {name}")
        model_report = {
            f"{name}": {
                "epoch_stop": state['epoch'],
                "mode": experiment_config['eval_config'].get('mode', 'plain') + str(
                    experiment_config.get('threshold', '')),
                "add_info": experiment_config['data_config'].get('add_info', None)
            }
        }

        # Dev data
        if config_or_modelpath != 'debug.json':
            prediction_dev_df, experiment_config = predict(model, tokenizer, experiment_config, "eval")
            golden_dev, pred_dev, match_prob_dev = prediction_dev_df.golden_label, prediction_dev_df.prediction, prediction_dev_df.score

            model_report[name].update(evaluate(pred_dev, golden_dev, match_prob_dev))

        # Test data

        model_report[name].update({"model_path": str(model_path)})
        final_report[fold_i] = model_report
    return final_report, folds_predictions


# ============================================
# Performance Report
# ============================================

def report_kfold(folds_reports: Dict, report_path=None, folds_predictions: List[pd.DataFrame] = None):
    assert folds_reports is not None
    assert folds_predictions is not None

    model_report = _post_process_folds_report(folds_reports, folds_predictions)

    if report_path is None:
        report_path = Path('.', "report.csv")
    if Path(report_path).is_file():
        _tmp_report_df = pd.read_csv(report_path, index_col='name')
        report_dict = _tmp_report_df.to_dict('index')
    else:
        report_dict = {}

    report_dict.update(model_report)
    report_df = pd.DataFrame.from_dict(report_dict, orient='index').fillna(-1)
    report_df.index.name = 'name'
    report_df.to_csv(report_path, float_format='%.5f')


def _post_process_folds_report(reports, predictions) -> Dict:
    prediction_df = predictions[0][['arg_id', 'key_point_id', 'prediction', 'golden_label']]
    scores = [predictions[i]['score'] for i in range(len(predictions))]
    scores = sum(scores) / len(predictions)
    prediction_df['score'] = scores

    submission_file_path = "./predictions/" + 'folds' + ".json"
    generate_submission(prediction_df, submission_file_path)
    mAP_strict, mAP_relaxed = calc_map(submission_file_path)

    report = reports[0]
    report_key = list(report.keys())[0] + '_kfold'

    report[report_key]['epoch_stop'] = -1
    report[report_key]['model_path'] = f"{len(predictions)} folds"

    metrics = ['acc_neg', 'acc_pos', 'acc', 'prec', 'recall', 'f1_pos', 'f1', 'auc']

    for i in range(1, len(predictions)):
        r = reports[i]
        for m in metrics:
            report[report_key][m] += r[report_key][m]

    for m in metrics:
        report[report_key][m] /= len(predictions)

    report[report_key]['mAP_strict'] = mAP_strict
    report[report_key]['mAP_relaxed'] = mAP_relaxed

    return report
