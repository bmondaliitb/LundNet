# This file is part of LundNet by F. Dreyer and H. Qu

import numpy as np
from scipy import sparse
from dgl import DGLGraph
from dgl import backend as F

'''copied from dgl.transform and fixed a bug'''

import torch

def pairwise_squared_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances.
    x: (N, D)
    returns: (N, N) where out[i,j] = ||x[i]-x[j]||^2
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x2 = (x * x).sum(dim=1, keepdim=True)          # (N,1)
    dist2 = x2 + x2.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))
    return torch.clamp(dist2, min=0.0)



def knn_graph(x, k):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix. The predecessors of each point are its k-nearest
    neighbors.

    If a 3D tensor is given instead, then each row would be transformed into
    a separate graph.  The graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.

        If 2D, each row of ``x`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)
    n_total_points = n_samples * n_points

    dist = pairwise_squared_distance(x)
    k_indices = F.argtopk(dist, k, 2, descending=False)
    dst = F.copy_to(k_indices, F.cpu())

    src = F.zeros_like(dst) + F.reshape(F.arange(0, n_points), (1, -1, 1))

    per_sample_offset = F.reshape(F.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))),
                            shape=(n_total_points, n_total_points))

    g = DGLGraph(adj, readonly=True)
    return g


def segmented_knn_graph(x, k, segs):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix.  The predecessors of each point are its k-nearest
    neighbors.

    The matrices are concatenated along the first axis, and are segmented by
    ``segs``.  Each block would be transformed into a separate graph.  The
    graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    k : int
        The number of neighbors
    segs : iterable of int
        Number of points of each point set.
        Must sum up to the number of rows in ``x``.

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    n_total_points, _ = F.shape(x)
    offset = np.insert(np.cumsum(segs), 0, 0)

    h_list = F.split(x, segs, 0)
    dst = [
        F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False) +
        offset[i]
        for i, h_g in enumerate(h_list)]
    dst = F.cat(dst, 0)
    src = F.arange(0, n_total_points).unsqueeze(1).expand(n_total_points, k)

    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    # !!! fix shape
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))),
                            shape=(n_total_points, n_total_points))

    g = DGLGraph(adj, readonly=True)
    return g


def reversed_graph(g):
    ret = DGLGraph()
    ret.add_nodes(g.number_of_nodes())
    u, v = g.all_edges()
    ret.add_edges(v, u)
    return ret
