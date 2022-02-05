import json
import os

import numpy as np
from sklearn.metrics import *

from track_1_kp_matching import get_predictions, evaluate_predictions
from utils import load_kpm_data


def evaluate(y_pred, y_true, score, suffix=''):
    neg_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 0]
    # neg_pos_prob = [score[i] for i, v in enumerate(y_true) if v == 0]
    neg_true = np.zeros(len(neg_pred), dtype=int)
    pos_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 1]
    # pos_pos_prob = [score[i] for i, v in enumerate(y_true) if v == 1]
    pos_true = np.ones(len(pos_pred), dtype=int)

    return {
        f"acc_neg{suffix}": accuracy_score(neg_true, neg_pred),
        f"acc_pos{suffix}": accuracy_score(pos_true, pos_pred),
        f"acc{suffix}": accuracy_score(y_true, y_pred),
        f"prec{suffix}": precision_score(y_true, y_pred),
        f"recall{suffix}": recall_score(y_true, y_pred),
        f"f1_pos{suffix}": f1_score(pos_true, pos_pred),
        f"f1{suffix}": f1_score(y_true, y_pred),
        f"auc{suffix}": roc_auc_score(y_true, score),
    }


def generate_submission(pred_df, output_file='./submission.csv'):
    submit_dict = {}
    for arg_id in set(pred_df.arg_id):
        _tmp_df = pred_df[pred_df.arg_id == arg_id][["key_point_id", "score"]].set_index("key_point_id")
        _tmp_dict = _tmp_df.to_dict()['score']
        submit_dict[arg_id] = _tmp_dict
    os.makedirs(os.path.dirname(str(output_file)), exist_ok=True)
    with open(output_file, 'w') as fp:
        json.dump(submit_dict, fp, indent=4, sort_keys=True)
    return output_file


def calc_map(submission_file):
    arg_df, kp_df, label_df = load_kpm_data(subset="test")
    df = get_predictions(submission_file, label_df, arg_df, kp_df)
    mAP_strict, mAP_relaxed = evaluate_predictions(df)
    return mAP_strict, mAP_relaxed
