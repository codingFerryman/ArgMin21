from math import sqrt

import numpy as np
import pandas as pd
from numpy import argmax
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from dataset_transformers import TransformersSentencePairDataset
from utils import get_data_path, get_logger

LOG_LEVEL = "INFO"


def load_model(path, model_class, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(pretrained_model, pretrained_tokenizer, config, subset="test"):
    assert subset in ["test", "dev"]

    # Set logger
    logger = get_logger("evaluation", level=LOG_LEVEL)

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eval_config = config["eval_config"]
    mode = eval_config.get("mode", "plain")
    tokenizer_config = config["tokenizer_config"]
    pretrained_model.to(device)
    pretrained_model.eval()
    logger.debug("Model prepared for evaluation")

    # Load data
    test_dataset = TransformersSentencePairDataset(tokenizer_config, subset, pretrained_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=eval_config.get("batch_size", 1))
    logger.debug("Data prepared for evaluation")

    # Predict
    kp_list = []
    arg_id_list = []
    predictions = []
    golden_labels = []
    probabilities = []
    with tqdm(test_dataloader, unit=' batch') as tdata:
        for batch in tdata:
            tdata.set_description(f"Prediction")
            arg_id_list.extend(batch.pop("arg_id"))
            kp_list.extend(batch.pop("key_point_id"))
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = pretrained_model(**batch)
            logits = outputs.logits
            _probs = torch.sigmoid(logits)
            true_prob = _probs[:, 1].cpu().detach().numpy()
            probabilities.extend(true_prob)
            _labels = batch['labels'].cpu().detach().numpy()
            golden_labels.extend(_labels)
            predictions.extend(torch.argmax(_probs, dim=1).cpu().detach().numpy())

    prediction_df = pd.DataFrame({
        "arg_id": arg_id_list,
        "key_point_id": kp_list,
        "prediction": predictions,
        "golden_label": golden_labels,
        "match_prob": probabilities
    })

    if "th" in mode:
        if "kp" not in mode:
            if subset == 'dev':
                fpr, tpr, thresholds = roc_curve(prediction_df.golden_label, prediction_df.match_prob, pos_label=1)
                ix = argmax(tpr - fpr)
                config['threshold'] = thresholds[ix]
                prediction_df['prediction'] = pd.Series(prediction_df.match_prob >= config['threshold']).astype(int)
            if subset == 'test':
                th = config['threshold']
                prediction_df['prediction'] = pd.Series(prediction_df.match_prob >= th).astype(int)
            if subset == 'test':
                _df_list = []
                for _arg_id in set(arg_id_list):
                    _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
                    _df['prediction'] = pd.Series(_df.match_prob >= config['threshold'][_arg_id]).astype(int)
                    _df_list.append(_df)
                prediction_df = pd.concat(_df_list)
        else:
            raise NotImplementedError
    elif "bm" in mode:
        for _arg_id in set(arg_id_list):
            _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
            _index = _df.sort_values('match_prob', ascending=False).index[0]
            prediction_df.at[_index, 'prediction'] = 1
    if "bmth" in mode:
        for _arg_id in set(arg_id_list):
            _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
            _index = _df.sort_values('match_prob', ascending=False).index[0]
            if prediction_df.loc[_index, 'match_prob'] >= config['threshold']:
                prediction_df.at[_index, 'prediction'] = 1
    return prediction_df, config


if __name__ == '__main__':
    from src.classifier_transformers import TransformersSentencePairClassifier

    model_c = load_model(
        "/home/he/Workspace/ArgMin21/models/bert-base BCELoss softmax_20211209-021638_state.pt/model_state.pt",
        TransformersSentencePairClassifier, config_path="/home/he/Workspace/ArgMin21/config/bert-base_best-match.json")
    result_c = predict(model_c)
