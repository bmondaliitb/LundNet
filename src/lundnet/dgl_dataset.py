# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import numpy as np

from torch.utils.data import Dataset
from .JetTree import JetTree, LundCoordinates
from .read_data import Jets
from .torch_graph import GraphData, batch_graphs, knn_edge_index, tree_edge_index
import torch
import torch.nn.functional as F
import sys
try:
    # `uproot3_methods` is not compatible with Python 3.12+.
    if sys.version_info >= (3, 12):
        raise ImportError
    from uproot_methods import TLorentzVectorArray, TLorentzVector
except Exception:
    try:
        if sys.version_info >= (3, 12):
            raise ImportError
        from uproot3_methods import TLorentzVectorArray, TLorentzVector
    except Exception:
        from .lorentz import TLorentzVectorArray, TLorentzVector
import time
import pandas as pd

groomer = None
dump_number_of_nodes = False


class TorchGraphDatasetLund(Dataset):

    fill_secondary = True
    node_coordinates = 'eta-phi'  # 'lund'

    def __init__(self, filepath_bkg, filepath_sig, nev=-1):
        super(TorchGraphDatasetLund, self).__init__()
        print('Start loading dataset %s (bkg) and %s (sig)' % (filepath_bkg, filepath_sig))
        tic = time.process_time()
        reader_bkg = Jets(filepath_bkg, nev, groomer=groomer)
        reader_sig = Jets(filepath_sig, nev, groomer=groomer)
        # attempt at using less memory
        self.data = []
        self.label = []
        for jet in reader_bkg:
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [0]
        for jet in reader_sig:
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [1]
        print(' ... Total time to read input files + construct the graphs for {num} jets: {ts} seconds'.format(
            num=len(self.label), ts=time.process_time() - tic))
        if dump_number_of_nodes:
            df = pd.DataFrame({'num_nodes': np.array(
                [g.number_of_nodes() for g in self.data]), 'label': np.array(self.label)})
            df.to_csv('num_nodes_lund_net_ktmin_%s_deltamin_%s.csv' % (JetTree.ktmin, JetTree.deltamin))
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def _build_tree(self, root):
        features = []
        coordinates = []
        edges = []
        jet_p4 = TLorentzVector(*root.node)

        def _rec_build(nid, node):
            branches = [node.harder, node.softer] if TorchGraphDatasetLund.fill_secondary else [node.harder]
            for branch in branches:
                if branch is None or branch.lundCoord is None:
                    # stop when reaching the leaf nodes
                    # we do not add the leaf nodes to the graph/tree as they do not have Lund coordinates
                    continue
                cid = len(features)
                if TorchGraphDatasetLund.node_coordinates == 'lund':
                    spatialCoord = branch.lundCoord.state()[:2]
                else:
                    node_p4 = TLorentzVector(*branch.node)
                    spatialCoord = np.array(
                        [delta_eta_reflect(node_p4, jet_p4),
                         node_p4.delta_phi(jet_p4)],
                        dtype='float32')
                coordinates.append(spatialCoord)
                features.append(branch.lundCoord.state())
                edges.append((cid, nid))
                _rec_build(cid, branch)
        # add root
        if root.lundCoord is not None:
            if TorchGraphDatasetLund.node_coordinates == 'lund':
                spatialCoord = root.lundCoord.state()[:2]
            else:
                spatialCoord = np.zeros(2, dtype='float32')
            coordinates.append(spatialCoord)
            features.append(root.lundCoord.state())
            _rec_build(0, root)
        else:
            # when a jet has only one particle (?)
            coordinates.append(np.zeros(2, dtype='float32'))
            features.append(np.zeros(LundCoordinates.dimension, dtype='float32'))
        ret = GraphData(
            features=np.stack(features, axis=0),
            coordinates=np.stack(coordinates, axis=0),
            edge_index=tree_edge_index(edges, len(features)),
        )
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].features.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


class TorchGraphDatasetParticle(Dataset):

    def __init__(self, filepath_bkg, filepath_sig, nev=-1):
        super(TorchGraphDatasetParticle, self).__init__()
        print('Start loading dataset %s (bkg) and %s (sig)' % (filepath_bkg, filepath_sig))
        tic = time.process_time()
        reader_bkg = Jets(filepath_bkg, nev, pseudojets=False, groomer=groomer)
        reader_sig = Jets(filepath_sig, nev, pseudojets=False, groomer=groomer)
        # Format of jets_bkg/jets_sig:
        # [# jet
        #     [ # particle
        #         [px, py, pz, E],
        #       # particle 2
        #         [px, py, pze, E]
        #     ],
        #  # jet 2
        #     #.....
        # ]
        # attempt at saving memory use:
        self.data = []
        self.label = []
        for constits in reader_bkg:
            self.data += [self._build_graph(constits)]
            self.label += [0]
        for constits in reader_sig:
            self.data += [self._build_graph(constits)]
            self.label += [1]
        print(' ... Total time to read input files + construct the graphs for {num} jets: {ts} seconds'.format(
            num=len(self.label), ts=time.process_time() - tic))
        if dump_number_of_nodes:
            df = pd.DataFrame({'num_nodes': np.array(
                [g.number_of_nodes() for g in self.data]), 'label': np.array(self.label)})
            df.to_csv('num_nodes_particle_net.csv')
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def _build_graph(self, constits):
        constits_p4 = TLorentzVectorArray.from_cartesian(*list(zip(*constits)))
        jet_p4 = constits_p4.sum()
        spatialCoord = np.stack([delta_eta_reflect(constits_p4, jet_p4), constits_p4.delta_phi(jet_p4)], axis=1)
        energyFeatures = np.log(np.stack([constits_p4.pt, constits_p4.energy], axis=1))
        features = np.concatenate([spatialCoord, energyFeatures], axis=1)
        ret = GraphData(features=features, coordinates=spatialCoord)
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].features.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


def delta_eta_reflect(constits_p4, jet_p4):
    deta = constits_p4.eta - jet_p4.eta
    return deta if jet_p4.eta > 0 else -deta


def pad_array(a, min_len=20, pad_value=0):
    if a.shape[0] < min_len:
        return F.pad(a, (0, 0, 0, min_len - a.shape[0]), mode='constant', value=pad_value)
    else:
        return a


class _SimpleCustomBatch:

    def __init__(self, data, k, min_nodes=20):
        transposed_data = list(zip(*data))
        graphs = []
        features = []
        for g in transposed_data[0]:
            n_nodes = max(g.number_of_nodes(), min_nodes)
            edge_index = knn_edge_index(g.coordinates, min(g.number_of_nodes() - 1, k))
            coords = pad_array(g.coordinates, min_nodes, 0)
            nng = GraphData(
                features=torch.empty((n_nodes, 0), dtype=torch.float32),
                coordinates=coords,
                edge_index=edge_index,
            )
            graphs.append(nng)
            fts = pad_array(g.features, min_nodes, 0)
            features.append(fts)
            assert(nng.number_of_nodes() == fts.shape[0])
        self.batch_graph = batch_graphs(graphs)
        self.features = torch.cat(features, 0)
        self.label = torch.tensor(transposed_data[1])

    def pin_memory(self):
        self.batch_graph = self.batch_graph.pin_memory()
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch, k):
    return _SimpleCustomBatch(batch, k)


class _LundTreeBatch:

    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.batch_graph = batch_graphs(transposed_data[0])
        self.features = torch.cat([g.features for g in transposed_data[0]], 0)
        self.label = torch.tensor(transposed_data[1])

    def pin_memory(self):
        self.batch_graph = self.batch_graph.pin_memory()
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper_tree(batch):
    return _LundTreeBatch(batch)


# Backwards-compatible names for older scripts that imported this module.
DGLGraphDatasetLund = TorchGraphDatasetLund
DGLGraphDatasetParticle = TorchGraphDatasetParticle
