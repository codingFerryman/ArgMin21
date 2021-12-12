import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

from dataset_transformers import TransformersSentencePairDataset
from utils import get_data_path, get_logger

LOG_LEVEL = "INFO"


def load_model(path, model_class, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, config, subset="test"):
    assert subset in ["test", "dev"]

    # Set logger
    logger = get_logger("evaluation", level=LOG_LEVEL)

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eval_config = config["eval_config"]
    mode = eval_config.get("mode", "plain")
    tokenizer_config = config["tokenizer_config"]
    model.to(device)
    model.eval()
    logger.debug("Model prepared for evaluation")

    # Load data
    test_dataset = TransformersSentencePairDataset(tokenizer_config, subset)
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
            # arg_id_list += batch.pop("arg_id")
            # kp_list += batch.pop("key_point_id")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            true_prob = logits[:, 1].cpu().detach().numpy()
            probabilities += [*true_prob]
            _labels = batch['labels'].cpu().detach().numpy()
            golden_labels += [*_labels]
            predict_labels = [*torch.argmax(logits, dim=1).cpu().detach().numpy()]
            predictions += predict_labels

    prediction_df = pd.DataFrame({
        # "arg_id": arg_id_list,
        # "key_point_id": kp_list,
        "prediction": predictions,
        "golden_label": golden_labels,
        "match_prob": probabilities
    })

    if "bm" in mode:
        for _arg_id in set(arg_id_list):
            _df = prediction_df[prediction_df.arg_id == _arg_id]
            if max(_df.prediction) == 0:
                _max_prob = max(_df.match_prob)
                prediction_df.loc[_df[_df.match_prob == _max_prob].index, 'prediction'] = 1
    return prediction_df


if __name__ == '__main__':
    from src.classifier_transformers import TransformersSentencePairClassifier
    model = load_model("/home/he/Workspace/ArgMin21/models/bert-base BCELoss softmax_20211209-021638_state.pt/model_state.pt", TransformersSentencePairClassifier, config_path="/home/he/Workspace/ArgMin21/config/bert-base_best-match.json")
    result = predict(model)
