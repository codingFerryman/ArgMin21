import json
import os
from pathlib import Path

import pandas as pd
import torch.cuda
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier import training
from evaluate import evaluate, generate_submission, calc_map
from predict import predict
from utils import get_project_path, get_logger, set_seed

set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = get_logger("main", "debug")

config_dir = Path(get_project_path(), 'config')


def run(config_or_modelpath, cuda_device="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    torch.cuda.empty_cache()
    logger.info(f"Model: {config_or_modelpath}")

    if config_or_modelpath[-5:] == '.json':
        config_path = Path(config_dir, config_or_modelpath)

        with open(config_path, 'r') as fc:
            experiment_config = json.load(fc)

        # ============================================
        # Train if configuration
        # ============================================

        trainer, model_path = training(config_path)
    else:

        # ============================================
        # Load if trained
        # ============================================

        model_path = config_or_modelpath
        with open(Path(model_path, 'training.json'), 'r') as fc:
            experiment_config = json.load(fc)

    name = experiment_config.get("name", "default")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(Path(model_path, 'state.json'), 'r') as fs:
        state = json.load(fs)

    # ============================================
    # Evaluate, predict, and report
    # ============================================

    # Report template
    logger.info(f"Preparing the report for {name}")
    model_report = {
        f"{name}": {
            "epoch_stop": state['epoch'],
            "mode": experiment_config['eval_config'].get('mode', 'plain') + str(experiment_config.get('threshold', '')),
            "add_info": experiment_config['data_config'].get('add_info', None)
        }
    }

    # Dev data
    if config_or_modelpath != 'debug.json':
        prediction_dev_df, experiment_config = predict(model, tokenizer, experiment_config, "eval")
        golden_dev, pred_dev, match_prob_dev = prediction_dev_df.golden_label, prediction_dev_df.prediction, prediction_dev_df.score

        model_report[name].update(evaluate(pred_dev, golden_dev, match_prob_dev))

    # Test data
    submission_file_path = "./predictions/" + name + ".json"
    prediction_test_df, experiment_config = predict(model, tokenizer, experiment_config, "test")
    generate_submission(prediction_test_df, submission_file_path)
    mAP_strict, mAP_relaxed = calc_map(submission_file_path)
    model_report[name].update(
        {
            "mAP_strict": mAP_strict,
            "mAP_relaxed": mAP_relaxed
        }
    )
    # golden_test, pred_test, match_prob_test = prediction_test_df.golden_label, prediction_test_df.prediction, prediction_test_df.score
    # tn_test, fp_test, fn_test, tp_test = confusion_matrix(golden_test, pred_test).ravel()

    # neg_pred = [pred_test[i] for i, v in enumerate(golden_test) if v == 0]
    # neg_pos_prob = [match_prob_test[i] for i, v in enumerate(golden_test) if v == 0]
    # neg_true = np.zeros(len(neg_pred), dtype=int)
    # pos_pred = [pred_test[i] for i, v in enumerate(golden_test) if v == 1]
    # pos_pos_prob = [match_prob_test[i] for i, v in enumerate(golden_test) if v == 1]
    # pos_true = np.ones(len(pos_pred), dtype=int)

    model_report[name].update({"model_path": str(model_path)})
    return model_report


# ============================================
# Performance Report
# ============================================

def report(model_report):
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
