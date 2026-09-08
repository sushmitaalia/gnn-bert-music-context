import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_classes=8):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.classifier(x)


class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SAGEConv(12, 64)
        self.conv2 = SAGEConv(64, 64)

    def forward(self, graph):
        x = F.relu(
            self.conv1(graph.x, graph.edge_index)
        )

        x = F.relu(
            self.conv2(x, graph.edge_index)
        )

        return global_mean_pool(
            x,
            graph.batch
        )


class AudioGNNEncoder(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=128):
        super().__init__()

        self.conv1 = SAGEConv(12, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph):
        x = F.relu(
            self.conv1(graph.x, graph.edge_index)
        )

        x = F.relu(
            self.conv2(x, graph.edge_index)
        )

        x = global_mean_pool(
            x,
            graph.batch
        )

        x = self.proj(x)

        return F.normalize(
            x,
            dim=1
        )
