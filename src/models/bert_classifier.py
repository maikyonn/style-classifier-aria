import torch
import torch.nn as nn
from transformers import BertModel
from typing import Tuple

class BertStyleClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.tokenizer_name)
        config.hidden_size = self.bert.config.hidden_size

        # Dropout layers
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.n_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout1(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)
        return self.dropout2(logits) 