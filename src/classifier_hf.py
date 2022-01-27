import json
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from typing import Union

from dataset_hf import TransformersSentencePairDataset
from utils import get_logger, get_project_path

LOG_LEVEL = "INFO"
logger = get_logger("training", level=LOG_LEVEL)


class KPMTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.], device=logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(*args, **kwargs):
    pass


def training(config_path: Union[str, Path]):
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load config and model
    assert Path(config_path).is_file(), f"This config file doesn't exist: {config_path}"
    with open(config_path, 'r') as fc:
        config = json.load(fc)
    name = config.get('name', 'default')
    output_path = Path(get_project_path(), "models", f"{name}_{now}")
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    trainer_config = config['trainer_config']
    tokenizer_config = config['tokenizer_config']
    assert tokenizer_config.get("type") == "transformer"
    model_config = config['model_config']
    assert model_config.get("type") == "transformer"
    model_name_or_path = model_config.get("name_or_path")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
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

    callbacks = [
        EarlyStoppingCallback(trainer_config.pop('early_stopping_patience', 5),
                              trainer_config.pop('early_stopping_threshold', 0.))
    ]

    training_args = TrainingArguments(
        output_dir=str(Path(output_path, 'checkpoints')),
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        # eval_accumulation_steps=10,
        load_best_model_at_end=True,
        fp16=True,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(output_path))
    trainer.save_state()
    trainer.state.save_to_json(str(Path(output_path, 'state.json')))
    train_dataset.tokenizer.save_pretrained(str(output_path))
    with open(Path(output_path, 'training.json'), 'w') as fc:
        json.dump(config, fc)
    return trainer, output_path
