import numpy as np
import pandas as pd
import scipy.special
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import KPADataset
from utils import get_logger, get_device

LOG_LEVEL = "INFO"

# Set logger
logger = get_logger("evaluation", level=LOG_LEVEL)


def load_model(path, model_class, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(pretrained_model, pretrained_tokenizer, config, subset="test"):
    if subset == 'dev':
        subset = 'eval'
    assert subset in ["test", "eval", "test_eval"]

    # Load model
    eval_config = config["eval_config"]
    tokenizer_config = config["tokenizer_config"]
    device = get_device()
    pretrained_model.to(device)
    pretrained_model.eval()
    logger.info(f"Model prepared for predicting {subset}")

    # Load data
    test_dataset = KPADataset(
        tokenizer_name=config['model_name'],
        tokenizer_config=tokenizer_config,
        subset=subset,
        pretrained_tokenizer=pretrained_tokenizer
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=eval_config.get("batch_size", 64))
    logger.debug("Data prepared for evaluation")

    # Predict

    kp_list = []
    arg_id_list = []
    predictions = []
    golden_labels = []
    probabilities = []
    with tqdm(test_dataloader, unit=' batch') as tdata:
        with torch.no_grad():
            for batch in tdata:
                tdata.set_description(f"Predicting {subset}")
                arg_id_list.extend(batch.pop("arg_id"))
                kp_list.extend(batch.pop("key_point_id"))

                batch = {k: v.to(device) for k, v in batch.items()}

                if subset != 'test':
                    _labels = batch['labels'].cpu().detach().numpy()
                else:
                    _labels = batch.pop('labels')

                outputs = pretrained_model(**batch)
                logits = outputs.logits
                _probs = scipy.special.softmax(scipy.special.softmax(logits.detach().cpu().numpy()), axis=1)
                true_prob = _probs[:, 1]
                probabilities.extend(true_prob)

                golden_labels.extend(_labels)
                predictions.extend(np.argmax(_probs, axis=1))

    prediction_df = pd.DataFrame({
        "arg_id": arg_id_list,
        "key_point_id": kp_list,
        "prediction": predictions,
        "golden_label": golden_labels,
        "score": probabilities
    })

    return prediction_df, config

    # mode = eval_config.get("mode", "simple")
    # if ("th" in mode) and ("bm" not in mode):
    #     if subset == 'dev':
    #         if "f1" in mode:
    #             thresholds_space = np.logspace(-1., 0., 5000)
    #             f1_max = 0.
    #             config_threshold = -1.
    #             for th in thresholds_space:
    #                 _predtions = pd.Series(prediction_df.score >= th).astype(int)
    #                 _f1 = f1_score(prediction_df.golden_label, _predtions)
    #                 if _f1 > f1_max:
    #                     f1_max = _f1
    #                     config_threshold = th
    #                     prediction_df['prediction'] = _predtions
    #             config['threshold'] = config_threshold
    #         else:
    #             fpr, tpr, thresholds = roc_curve(prediction_df.golden_label, prediction_df.score, pos_label=1)
    #             ix = argmax(tpr - fpr)
    #             config['threshold'] = thresholds[ix]
    #             prediction_df['prediction'] = pd.Series(prediction_df.score >= config['threshold']).astype(int)
    #     if subset == 'test':
    #         th = config['threshold']
    #         prediction_df['prediction'] = pd.Series(prediction_df.score >= th).astype(int)
    # elif ("bm" in mode) and ("th" not in mode):
    #     for _arg_id in set(arg_id_list):
    #         _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
    #         _index = _df.sort_values('score', ascending=False).index[0]
    #         prediction_df.at[_index, 'prediction'] = 1
    # elif "bmth" in mode:
    #     if subset == 'dev':
    #         thresholds_space = np.linspace(0., 0.1, 100)
    #         f1_max = 0.
    #         config_threshold = -1.
    #         _predictions = prediction_df['prediction'].copy()
    #         for th in tqdm(thresholds_space):
    #             for _arg_id in set(arg_id_list):
    #                 _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
    #                 _index = _df.sort_values('score', ascending=False).index[0]
    #                 _prob = _df.loc[_index, 'score']
    #                 if _prob >= th:
    #                     prediction_df.at[_index, 'prediction'] = 1
    #             _f1 = f1_score(prediction_df.golden_label, prediction_df.prediction)
    #             if _f1 > f1_max:
    #                 f1_max = _f1
    #                 config_threshold = th
    #                 _predictions = prediction_df['prediction'].copy()
    #         prediction_df['prediction'] = _predictions
    #         config['threshold'] = config_threshold
    #     if subset == 'test':
    #         th = config['threshold']
    #         for _arg_id in set(arg_id_list):
    #             _df = prediction_df[prediction_df.arg_id == _arg_id].copy()
    #             _index = _df.sort_values('score', ascending=False).index[0]
    #             _prob = _df.loc[_index, 'score']
    #             if _prob >= th:
    #                 prediction_df.at[_index, 'prediction'] = 1

# if __name__ == '__main__':
#     from src.classifier_transformers import TransformersSentencePairClassifier
#
#     model_c = load_model(
#         "/home/he/Workspace/ArgMin21/models/bert-base.json BCELoss softmax_20211209-021638_state.pt/model_state.pt",
#         TransformersSentencePairClassifier, config_path="/home/he/Workspace/ArgMin21/config/bert-base_best-match.json")
#     result_c = predict(model_c)
