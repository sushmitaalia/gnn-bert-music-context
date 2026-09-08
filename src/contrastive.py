import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import AudioGNNEncoder
from bert_encoder import CaptionBERTEncoder


class Task4ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.audio_encoder = AudioGNNEncoder(
            hidden_dim=128,
            out_dim=embedding_dim
        )

        self.text_encoder = CaptionBERTEncoder(
            out_dim=embedding_dim
        )

    def forward(
        self,
        graph,
        input_ids,
        attention_mask
    ):
        audio_embedding = self.audio_encoder(
            graph
        )

        text_embedding = self.text_encoder(
            input_ids,
            attention_mask
        )

        return audio_embedding, text_embedding


def contrastive_loss(
    audio_embeddings,
    text_embeddings,
    temperature=0.07
):
    logits = audio_embeddings @ text_embeddings.T
    logits = logits / temperature

    labels = torch.arange(
        logits.size(0),
        device=logits.device
    )

    audio_to_text_loss = F.cross_entropy(
        logits,
        labels
    )

    text_to_audio_loss = F.cross_entropy(
        logits.T,
        labels
    )

    return (
        audio_to_text_loss + text_to_audio_loss
    ) / 2
