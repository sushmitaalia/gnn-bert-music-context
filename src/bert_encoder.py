import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODEL_NAME = "distilbert-base-uncased"


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask, return_tokens=False):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if return_tokens:
            return out.last_hidden_state

        return out.last_hidden_state[:, 0, :]


class CaptionBERTEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.proj = nn.Linear(
            self.bert.config.hidden_size,
            out_dim
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = output.last_hidden_state[:, 0]

        text_embedding = self.proj(
            text_embedding
        )

        return F.normalize(
            text_embedding,
            dim=1
        )
