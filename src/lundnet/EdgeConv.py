# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import torch
import torch.nn as nn


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer implemented with plain PyTorch tensors.

    The graph argument is a lundnet.torch_graph.GraphBatch with edge_index in
    the common [source, destination] format.
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

    def forward(self, graph, h):
        if graph.edge_index.numel() == 0:
            x = h.new_zeros((h.shape[0], self.theta.out_features))
        else:
            src, dst = graph.edge_index
            msg = self.theta(h[dst] - h[src]) + self.phi(h[src])
            for i in range(self.num_layers):
                if i > 0:
                    msg = self.fcs[i - 1](msg)
                if self.batch_norm:
                    msg = self.bns[i](msg)
                if self.activation:
                    msg = self.acts[i](msg)

            x = h.new_zeros((h.shape[0], msg.shape[1]))
            x.index_add_(0, dst, msg)
            deg = torch.bincount(dst, minlength=h.shape[0]).to(h.device).clamp(min=1).float()
            x = x / deg.unsqueeze(1)

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
