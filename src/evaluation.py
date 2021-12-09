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


def predict(model, subset="test"):
    assert subset in ["test", "dev"]
    # Set logger
    logger = get_logger("evaluation", level=LOG_LEVEL)

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eval_config = model.get_eval_config()
    tokenizer_config = model.get_tokenizer_config()
    model.to(device)
    model.eval()
    logger.debug("Model prepared for evaluation")

    # Load data
    test_dataset = TransformersSentencePairDataset(tokenizer_config, subset)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=eval_config.get("batch_size", 1))
    logger.debug("Data prepared for evaluation")

    # Predict
    predictions = []
    golden_labels = []
    probabilities = []
    with tqdm(test_dataloader, unit=' batch') as tdata:
        for batch in tdata:
            tdata.set_description(f"Prediction")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            true_prob = logits[:, 1].cpu().detach().numpy()
            probabilities += [*true_prob]
            _labels = batch['labels'].cpu().detach().numpy()
            golden_labels += [*_labels]
            predict_labels = [*torch.argmax(logits, dim=1).cpu().detach().numpy()]
            predictions += predict_labels
    return predictions, golden_labels, probabilities

#
# if __name__ == '__main__':
#     from src.classifier_transformers import TransformersSentencePairClassifier
#     model = load_model("/home/he/Workspace/ArgMin21/models/bert-base-uncased_20211208-222730/model_state.pt", TransformersSentencePairClassifier, config_path="/config/bert-base_sigmoid.json")
#     predictions, golden_labels, probabilities = predict(model)
