import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class TransformersSentencePairClassifier(nn.Module):

    def __init__(self, model_name_or_path, num_labels=2, freeze_model=False):
        if 'bert' not in model_name_or_path.lower():
            raise NotImplementedError("Only support BERT-based models at this time.")
        if num_labels != 2:
            raise NotImplementedError("parameter \"num_labels\" only supports 2")
        self.num_labels = num_labels

        super(TransformersSentencePairClassifier, self).__init__()

        self.model = AutoModel.from_pretrained(model_name_or_path)

        if "albert-base-v2" in model_name_or_path:
            hidden_size = 768
        elif "albert-large-v2" in model_name_or_path:
            hidden_size = 1024
        elif "albert-xlarge-v2" in model_name_or_path:
            hidden_size = 2048
        elif "albert-xxlarge-v2" in model_name_or_path:
            hidden_size = 4096
        elif "bert-base-cased" in model_name_or_path:
            hidden_size = 768
        else:
            raise NotImplementedError("The model {model_name_or_path} is not supported.")

        # Only train the classification layer weights
        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.dropout = nn.Dropout(
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids
                             )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
