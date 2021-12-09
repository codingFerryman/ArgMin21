import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment import MyExperiment
from transformers.modeling_outputs import SequenceClassifierOutput
from pathlib import Path
from config_map import model_map


class TransformersSentencePairClassifier(MyExperiment):

    def __init__(self, config_path, num_labels=2):

        self.num_labels = num_labels

        super(TransformersSentencePairClassifier, self).__init__(config_path)
        model_name_or_path = self.model_config['name_or_path']
        if Path(model_name_or_path).is_file():
            self.model = torch.load(model_name_or_path)
        else:
            self.model = model_map(self.model_config)

        # Activation
        self.activation = self.trainer_config.get('activation', 'sigmoid')

        # Only train the classification layer weights
        is_frozen = self.trainer_config.get('freeze', False)
        if is_frozen is True:
            for p in self.model.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids
                             )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.activation == 'sigmoid':
            logits = torch.sigmoid(logits)
        elif self.activation == 'softmax':
            logits = torch.softmax(logits, dim=1)
        else:
            raise NotImplementedError
        loss_fct = eval(self.trainer_config['loss'])()

        labels_one_hot = F.one_hot(labels, num_classes=self.num_labels)
        loss = loss_fct(logits, labels_one_hot.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
