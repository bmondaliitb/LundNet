# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl.function as fn
import torch.nn as nn
from contextlib import contextmanager


def _fn_copy_e(src_field: str, out_field: str):
    # DGL renamed copy_e -> copy_edge in older versions.
    if hasattr(fn, 'copy_e'):
        return fn.copy_e(src_field, out_field)
    if hasattr(fn, 'copy_edge'):
        return fn.copy_edge(src_field, out_field)
    raise AttributeError("dgl.function has no copy_e/copy_edge")


def _fn_mean(src_field: str, out_field: str):
    # Deprecated: kept for backwards compatibility with earlier patch.
    if hasattr(fn, 'mean'):
        return fn.mean(src_field, out_field)
    return None


def _fn_sum(src_field: str, out_field: str):
    if hasattr(fn, 'sum'):
        return fn.sum(src_field, out_field)
    raise AttributeError("dgl.function has no sum")


def _in_degrees(g):
    if hasattr(g, 'in_degrees'):
        return g.in_degrees()
    # Very old DGL sometimes exposes in_degree(u)
    if hasattr(g, 'in_degree'):
        try:
            return g.in_degree(g.nodes())
        except Exception:
            return g.in_degree()
    raise AttributeError("DGL graph has no in_degrees/in_degree")


def _update_all_mean(lg, edge_field: str, out_field: str):
    """Run update_all with mean aggregation across DGL versions."""
    mean_reduce = _fn_mean(edge_field, out_field)
    if mean_reduce is not None:
        lg.update_all(_fn_copy_e(edge_field, edge_field), mean_reduce)
        return

    # Fallback: sum then divide by in-degree.
    msg_field = '__m'
    lg.update_all(_fn_copy_e(edge_field, msg_field), _fn_sum(msg_field, out_field))
    deg = _in_degrees(lg)
    try:
        deg = deg.to(lg.ndata[out_field].device)
    except Exception:
        pass
    if len(deg.shape) == 1:
        deg = deg.unsqueeze(1)
    deg = deg.clamp(min=1).float()
    lg.ndata[out_field] = lg.ndata[out_field] / deg
    # clean up temp edge field when possible
    try:
        lg.edata.pop(msg_field)
    except Exception:
        pass


@contextmanager
def _local_graph_scope(g):
    """Provide a local-scope graph across DGL versions.

    - Newer DGL: `with g.local_scope(): ...`
    - Older DGL: `g = g.local_var()`
    - Fallback: manually restore ndata/edata
    """
    if hasattr(g, 'local_scope'):
        with g.local_scope():
            yield g
        return

    if hasattr(g, 'local_var'):
        yield g.local_var()
        return

    ndata_backup = dict(getattr(g, 'ndata', {}))
    edata_backup = dict(getattr(g, 'edata', {}))
    try:
        yield g
    finally:
        if hasattr(g, 'ndata'):
            g.ndata.clear()
            g.ndata.update(ndata_backup)
        if hasattr(g, 'edata'):
            g.edata.clear()
            g.edata.update(edata_backup)


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "Dynamic Graph CNN for Learning on Point Clouds" (https://arxiv.org/pdf/1801.07829).
    Code adapted from https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/edgeconv.py.
    """

    def __init__(self, in_feat, out_feats, batch_norm=True, activation=True):
        super(EdgeConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)

        out_feat = out_feats[0]
        self.theta = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.phi = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.fcs = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.fcs.append(nn.Linear(out_feats[i - 1], out_feats[i], bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm1d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Linear(in_feat, out_feats[-1], bias=False if self.batch_norm else True)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def message(self, edges):
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        with _local_graph_scope(g) as lg:
            lg.ndata['x'] = h
            # generate the message and store it on the edges
            lg.apply_edges(self.message)
            # process the message
            e = lg.edata['e']
            for i in range(self.num_layers):
                if i > 0:
                    e = self.fcs[i - 1](e)
                if self.batch_norm:
                    e = self.bns[i](e)
                if self.activation:
                    e = self.acts[i](e)
            lg.edata['e'] = e
            # pass the message and update the nodes
            _update_all_mean(lg, edge_field='e', out_field='x')
            # shortcut connection
            x = lg.ndata.pop('x')
            lg.edata.pop('e')
            if self.sc is None:
                sc = h
            else:
                sc = self.sc(h)
                if self.batch_norm:
                    sc = self.sc_bn(sc)
            if self.activation:
                return self.sc_act(x + sc)
            else:
                return x + sc
