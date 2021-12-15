import json
from pathlib import Path
from utils import manual_seed
from typing import Union
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from dataset_transformers import TransformersSentencePairDataset
from utils import get_logger, get_project_path

LOG_LEVEL = "INFO"
logger = get_logger("training", level=LOG_LEVEL)


class MyTrainer(Trainer):
    def __init__(self, loss_fct, activation_fct, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = loss_fct
        self.activation_fct = activation_fct

    def compute_loss(self, model, inputs, **kwargs):
        labels = inputs.pop("labels")
        labels_one_hot = F.one_hot(labels, num_classes=2)
        outputs = model(**inputs)
        logits = outputs[0]
        if self.activation_fct == 'sigmoid':
            logits = torch.sigmoid(logits).float()
        elif self.activation_fct == 'softmax':
            logits = torch.softmax(logits, dim=1).float()
        elif self.activation_fct == 'log_softmax':
            logits = torch.log_softmax(logits, dim=1).float()
        elif self.activation_fct == 'gelu':
            logits = torch.nn.functional.gelu(logits).float()
        else:
            raise NotImplementedError

        loss_fct = eval(self.loss_fct)()
        loss = loss_fct(logits, labels_one_hot.float())
        return (loss, outputs) if kwargs.get("return_outputs") else loss


def training(config_path: Union[str, Path], val_ratio=0.01):
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Set logger

    # Load config and model
    assert Path(config_path).is_file(), f"This config file doesn't exist: {config_path}"
    with open(config_path, 'r') as fc:
        config = json.load(fc)
    name = config.get('name', 'default')
    output_path = Path(get_project_path(), "models", f"{name}_{now}")
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    seed = config.get('seed', None)
    manual_seed(seed)
    trainer_config = config['trainer_config']
    tokenizer_config = config['tokenizer_config']
    assert tokenizer_config.get("type") == "transformer"
    model_config = config['model_config']
    assert model_config.get("type") == "transformer"
    model_name_or_path = model_config.get("name_or_path")

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               num_labels=2,
                                                               **model_config.get("args")
                                                               )

    train_dataset = TransformersSentencePairDataset(
        tokenizer_config=tokenizer_config,
        subset="train"
    )

    val_dataset = TransformersSentencePairDataset(
        tokenizer_config=tokenizer_config,
        subset="dev"
    )
    #
    # val_size = int(val_ratio*len(train_dataset))
    # train_size = len(train_dataset) - val_size
    # train_dataset, val_dataset = random_split(train_dataset,
    #                                           [train_size, val_size],
    #                                           generator=torch.Generator().manual_seed(seed))

    callbacks = [
        EarlyStoppingCallback(trainer_config.pop('early_stopping_patience', 5),
                              trainer_config.pop('early_stopping_threshold', 0.))
    ]

    loss_fct = trainer_config.pop("loss_fct", "torch.nn.BCEWithLogitsLoss")
    activation_fct = trainer_config.pop("activation_fct", "torch.sigmoid")

    training_args = TrainingArguments(
        output_dir=str(Path(output_path, 'checkpoints')),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        # greater_is_better=True,
        fp16=True,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # loss_fct=loss_fct,
        # activation_fct=activation_fct,
        eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    trainer.save_model(str(output_path))
    train_dataset.tokenizer.save_pretrained(str(output_path))
    with open(Path(output_path, 'training.json'), 'w') as fc:
        json.dump(config, fc)
    return trainer, output_path


# def compute_metrics(eval_pred, metrics="accuracy,f1"):
#     metric = load_metric("f1")
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     result = metric.compute(predictions=predictions, references=labels)
#     return result
#     # metric_list = metrics.split(',')
#     # metric_list = [metric.strip() for metric in metric_list]
#     # result = {}
#     # for metric in metric_list:
#     #     metric_c = load_metric(metric)
#     #     logits, labels = eval_pred
#     #     predictions = np.argmax(logits, axis=-1)
#     #     result[metric] = metric_c.compute(predictions=predictions, references=labels)
#     # return result
