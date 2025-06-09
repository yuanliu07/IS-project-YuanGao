import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(self, n_classes, pretrained_model_path):
        super(TransformerModel, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_path)
        self.drop = nn.Dropout(p=0.1)
        # self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = outputs.last_hidden_state

        # pooler_output = outputs.pooler_output
        # output = self.drop(output)
        return output
