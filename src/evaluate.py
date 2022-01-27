import json

import numpy as np
from sklearn.metrics import *

from track_1_kp_matching import get_predictions, evaluate_predictions
from utils import load_kpm_data


def evaluate(y_pred, y_true, score):
    neg_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 0]
    # neg_pos_prob = [score[i] for i, v in enumerate(y_true) if v == 0]
    neg_true = np.zeros(len(neg_pred), dtype=int)
    pos_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 1]
    # pos_pos_prob = [score[i] for i, v in enumerate(y_true) if v == 1]
    pos_true = np.ones(len(pos_pred), dtype=int)

    return {
        "acc_neg": accuracy_score(neg_true, neg_pred),
        "acc_pos": accuracy_score(pos_true, pos_pred),
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_pos": f1_score(pos_true, pos_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, score),
    }


def generate_submission(pred_df, output_file='./submission.csv'):
    submit_dict = {}
    for arg_id in set(pred_df.arg_id):
        _tmp_df = pred_df[pred_df.arg_id == arg_id][["key_point_id", "score"]].set_index("key_point_id")
        _tmp_dict = _tmp_df.to_dict()['score']
        submit_dict[arg_id] = _tmp_dict
    with open(output_file, 'w') as fp:
        json.dump(submit_dict, fp, indent=4, sort_keys=True)
    return output_file


def calc_map(submission_file):
    arg_df, kp_df, label_df = load_kpm_data(subset="test")
    df = get_predictions(submission_file, label_df, arg_df, kp_df)
    mAP_strict, mAP_relaxed = evaluate_predictions(df)
    return mAP_strict, mAP_relaxed
