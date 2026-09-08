import torch
import torch.nn as nn

from gnn_model import GNNEncoder
from bert_encoder import TextEncoder


class Task3Model(nn.Module):
    def __init__(self, mode, num_labels=10):
        super().__init__()

        self.mode = mode
        self.gnn = GNNEncoder()
        self.text = TextEncoder()

        if mode == "bert":
            self.classifier = nn.Linear(768, num_labels)

        elif mode == "gnn":
            self.classifier = nn.Linear(64, num_labels)

        elif mode == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(832, 256),
                nn.ReLU(),
                nn.Linear(256, num_labels)
            )

        elif mode == "cross":
            self.graph_proj = nn.Linear(64, 256)
            self.text_proj = nn.Linear(768, 256)

            self.attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=4,
                batch_first=True
            )

            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU()
            )

            self.classifier = nn.Linear(
                256,
                num_labels
            )

    def forward(self, graph, input_ids, attention_mask):

        if self.mode == "bert":
            text_embedding = self.text(
                input_ids,
                attention_mask
            )

            return self.classifier(
                text_embedding
            )

        if self.mode == "gnn":
            graph_embedding = self.gnn(
                graph
            )

            return self.classifier(
                graph_embedding
            )

        if self.mode == "concat":
            graph_embedding = self.gnn(
                graph
            )

            text_embedding = self.text(
                input_ids,
                attention_mask
            )

            fused = torch.cat(
                [
                    graph_embedding,
                    text_embedding
                ],
                dim=1
            )

            return self.classifier(
                fused
            )

        graph_embedding = self.gnn(
            graph
        )

        graph_embedding = self.graph_proj(
            graph_embedding
        )

        text_tokens = self.text(
            input_ids,
            attention_mask,
            return_tokens=True
        )

        text_tokens = self.text_proj(
            text_tokens
        )

        attended_text, _ = self.attention(
            query=graph_embedding.unsqueeze(1),
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~attention_mask.bool()
        )

        attended_text = attended_text.squeeze(1)

        z = self.fusion(
            torch.cat(
                [
                    graph_embedding,
                    attended_text
                ],
                dim=1
            )
        )

        return self.classifier(z)
