from __future__ import annotations

import torch


class GraphData:
    """Single graph stored as plain torch tensors."""

    def __init__(self, features, coordinates, edge_index=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    def number_of_nodes(self):
        return self.features.shape[0]


class GraphBatch:
    """Batched homogeneous graph using PyTorch-style edge_index tensors."""

    def __init__(self, edge_index, batch, num_nodes_per_graph):
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.batch = torch.as_tensor(batch, dtype=torch.long)
        self.num_nodes_per_graph = torch.as_tensor(num_nodes_per_graph, dtype=torch.long)

    @property
    def num_nodes(self):
        return int(self.batch.numel())

    def batch_num_nodes(self):
        return self.num_nodes_per_graph

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.num_nodes_per_graph = self.num_nodes_per_graph.to(device)
        return self

    def pin_memory(self):
        self.edge_index = self.edge_index.pin_memory()
        self.batch = self.batch.pin_memory()
        self.num_nodes_per_graph = self.num_nodes_per_graph.pin_memory()
        return self


def batch_graphs(graphs):
    edge_indices = []
    batch = []
    counts = []
    offset = 0
    for graph_idx, graph in enumerate(graphs):
        n_nodes = graph.number_of_nodes()
        counts.append(n_nodes)
        batch.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        if graph.edge_index.numel() > 0:
            edge_indices.append(graph.edge_index + offset)
        offset += n_nodes

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return GraphBatch(edge_index=edge_index, batch=torch.cat(batch), num_nodes_per_graph=counts)


def mean_pool(x, batch, num_graphs=None):
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out = x.new_zeros((num_graphs, x.shape[1]))
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).to(x.device).clamp(min=1)
    return out / counts.unsqueeze(1)


def tree_edge_index(edges, num_nodes):
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    directed_edges = []
    for src, dst in edges:
        directed_edges.append((src, dst))
        directed_edges.append((dst, src))
    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def knn_edge_index(x, k, segments=None, include_self=False):
    """Build directed kNN edges where neighbors send messages to each node."""
    if segments is None:
        segments = [x.shape[0]]
    edges = []
    offset = 0
    for n_nodes in segments:
        if n_nodes <= 0:
            continue
        xs = x[offset:offset + n_nodes]
        k_eff = min(k + (0 if include_self else 1), n_nodes)
        dist = torch.cdist(xs, xs, p=2)
        nn_idx = dist.topk(k_eff, largest=False).indices
        dst = torch.arange(n_nodes, device=x.device).unsqueeze(1).expand_as(nn_idx)
        if not include_self:
            keep = nn_idx != dst
            nn_idx = nn_idx[keep].view(n_nodes, -1)
            dst = dst[keep].view(n_nodes, -1)
            if nn_idx.shape[1] > k:
                nn_idx = nn_idx[:, :k]
                dst = dst[:, :k]
        src = nn_idx.reshape(-1) + offset
        dst = dst.reshape(-1) + offset
        if src.numel() > 0:
            edges.append(torch.stack([src, dst], dim=0))
        offset += n_nodes

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edges, dim=1)
