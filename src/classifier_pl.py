from typing import Optional, Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import datasets
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.optim import Optimizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, \
    AutoModel
from tqdm import tqdm

from src.dataset_pl import KPMDataModule
from utils import set_seed

AVAIL_GPUS = min(1, torch.cuda.device_count())


class LitProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        _bar = tqdm(
            disable=True,
        )
        return _bar


class KPMClassifier(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            task_name: str,
            num_labels: int = 1,
            nr_frozen_epochs: int = 0,
            learning_rate: float = 5e-4,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.1,
            train_batch_size: int = 32,
            eval_batch_size: int = 16,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        print(self.hparams)
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=num_labels,
        )

        self.model = AutoModel.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     self.hparams.model_name_or_path,
        #     config=self.config
        # )

        classifier_dropout = (
            self.model.config.classifier_dropout if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        self.pos_threshold = kwargs.get('pos_threshold', 0.5)
        # self.pos_weight = kwargs.get('pos_weight', None)
        # # self.loss_fct = nn.BCEWithLogitsLoss(
        # #     # pos_weight=torch.tensor([self.pos_weight]),
        # #     pos_weight=torch.tensor([4]),
        # # )
        # self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.], device=self.device),)

    def on_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()
        print('\n')

    def forward(self, **inputs) -> Any:
        labels = inputs.pop('labels', None)
        outputs = self.model(**inputs)
        outputs = nn.Linear(self.model.config.hidden_size, 256).cuda()(outputs.pooler_output)
        outputs = self.dropout(outputs)
        outputs = nn.Linear(256, 128).cuda()(outputs)
        outputs = self.dropout(outputs)
        outputs = nn.Linear(128, self.num_labels).cuda()(outputs)
        # loss = self.loss_fct(outputs, labels)
        # # loss = self.loss_fct(outputs.view(-1), labels.type(torch.cuda.FloatTensor))
        # return loss, outputs
        # outputs = self.model(**inputs)
        # outputs['logits'] = self.classifier(self.dropout(outputs.pooler_output)).view(-1)

        loss = torch.nn.BCEWithLogitsLoss()(
            outputs.view(-1),
            labels.type(torch.cuda.FloatTensor)
        )
        # pt = torch.exp(-BCE_loss)
        # F_loss = 0.25 * (1-pt)**2 * BCE_loss
        # outputs['loss'] = torch.mean(F_loss)
        return loss, outputs

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        pass

    def training_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_loss, outputs = self(**batch)
        self.log('val_loss', val_loss, prog_bar=True)

        logits = outputs
        logits = torch.sigmoid(logits).view(-1)

        preds = torch.zeros(logits.shape)
        preds[logits.ge(self.pos_threshold)] = 1.
        # preds = torch.argmax(logits, axis=1)

        labels = batch["labels"]

        return {"val_loss": val_loss, "preds": preds, "labels": labels, "logits": logits}

    def validation_epoch_end(self, outputs):

        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        # pos_probs = torch.cat([torch.softmax(x["logits"], dim=1)[:, 1] for x in outputs]).detach().cpu().numpy()
        pos_probs = torch.cat([x["logits"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        self.log_dict(self.compute_classified_metrics(labels, preds, pos_probs), prog_bar=True)

        tensorboard_logs = {'val_avg_loss': loss}

        return {"val_loss": loss, 'log': tensorboard_logs}

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        if self.hparams.nr_frozen_epochs == 0 or self._frozen:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = True
        self._frozen = False

    @staticmethod
    def compute_classified_metrics(y_true, y_pred, y_pos_prob) -> Dict:
        neg_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 0]
        # neg_pos_prob = [y_pos_prob[i] for i, v in enumerate(y_true) if v == 0]
        neg_true = np.zeros(len(neg_pred), dtype=int)
        pos_pred = [y_pred[i] for i, v in enumerate(y_true) if v == 1]
        # pos_pos_prob = [y_pos_prob[i] for i, v in enumerate(y_true) if v == 1]
        pos_true = np.ones(len(pos_pred), dtype=int)
        return {
            "neg_acc": accuracy_score(neg_true, neg_pred),
            # "neg_f1": f1_score(neg_true, neg_pred),
            "pos_acc": accuracy_score(pos_true, pos_pred),
            # "pos_f1": f1_score(pos_true, pos_pred),
            "all_acc": accuracy_score(y_true, y_pred),
            "all_f1": f1_score(y_true, y_pred),
            "all_auc": roc_auc_score(y_true, y_pos_prob)
        }
