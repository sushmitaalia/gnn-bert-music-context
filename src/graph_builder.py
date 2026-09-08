import torch
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.997


def build_temporal_edges(num_nodes):
    edges = []

    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()


def build_similarity_edges(features, threshold=SIM_THRESHOLD):
    similarities = cosine_similarity(features)
    edges = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if similarities[i, j] >= threshold:
                edges.append([i, j])
                edges.append([j, i])

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()
