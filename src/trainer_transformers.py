import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from datasets import load_metric

from dataset_transformers import TransformersSentencePairDataset
from classifier_transformers import TransformersSentencePairClassifier


def trainer(train_data: Dataset,
            dev_data: Dataset,
            model_name_or_path: str,
            batch_size=64,
            num_epochs=10
            ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    model = TransformersSentencePairClassifier(model_name_or_path, freeze_model=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute()) # 0.7865818392134182


if __name__ == '__main__':
    _model = "bert-base-cased"
    max_len = 128

    train_dataset = TransformersSentencePairDataset(_model, max_len, "train")
    dev_dataset = TransformersSentencePairDataset(_model, max_len, "dev")

    trainer(train_dataset, dev_dataset, _model, num_epochs=3, batch_size=128)
