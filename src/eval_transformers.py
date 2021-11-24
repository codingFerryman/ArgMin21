import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score

from src.dataset_transformers import TransformersSentencePairDataset
from utils import get_data_path
from data.code.track_1_kp_matching import load_kpm_data, get_predictions, evaluate_predictions

test_df = pd.read_csv(Path(get_data_path(), 'test_data', 'labels_test.csv'))

model = torch.load("model.pt")
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

max_len = 128

test_data = TransformersSentencePairDataset("bert-base-uncased", max_len, "test")
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

predictions = []
for batch_ori in test_dataloader:
    batch = {k: v.to(device) for k, v in batch_ori.items()}
    output = model(**batch)
    prob = torch.nn.functional.softmax(output.logits, dim=1).cpu().detach().numpy()[0][1] # TODO Sigmoid: Output layer: 1
    label = int(torch.argmax(output.logits).cpu().detach().numpy())
    predictions.append(label)

prediction_df = test_df.copy()
prediction_df['label'] = predictions

prediction_df.to_csv('predictions.csv', index=False)

# prediction_df.to_json('predictions.json', index=False)
#
# arg_df, kp_df, labels_df = load_kpm_data(Path(get_data_path(), 'test_data'), subset="test")
#
# merged_df = get_predictions('predictions.csv', labels_df, arg_df, kp_df)
# evaluate_predictions(merged_df)