from typing import Union, Callable, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pprint import pprint

import networkx as nx
from torch_cluster import knn_graph, radius_graph

from mlreco.utils.metrics import *
from mlreco.utils.cluster.graph_batch import GraphBatch
from torch_geometric.data import Data as GraphData

from sklearn.neighbors import KNeighborsClassifier
from scipy.special import expit


def get_edge_weight(sp_emb: torch.Tensor,
                    ft_emb: torch.Tensor,
                    cov: torch.Tensor,
                    edge_indices: torch.Tensor,
                    occ=None,
                    eps=0.001):
    '''
    INPUTS:
        - sp_emb (N x D)
        - ft_emb (N x F)
        - cov (N x 2)
        - occ (N x 0)
        - edge_indices (2 x E)

    RETURNS:
        - pvec (N x 0)
    '''
    # print(edge_indices.shape)
    device = sp_emb.device
    ui, vi = edge_indices[0, :], edge_indices[1, :]

    sp_cov_i = cov[:, 0][ui]
    sp_cov_j = cov[:, 0][vi]
    sp_i = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_i**2 + eps)
    sp_j = ((sp_emb[ui] - sp_emb[vi])**2).sum(dim=1) / (sp_cov_j**2 + eps)

    ft_cov_i = cov[:, 1][ui]
    ft_cov_j = cov[:, 1][vi]
    ft_i = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_i**2 + eps)
    ft_j = ((ft_emb[ui] - ft_emb[vi])**2).sum(dim=1) / (ft_cov_j**2 + eps)

    p_ij = torch.exp(-sp_i-ft_i)
    p_ji = torch.exp(-sp_j-ft_j)

    pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1).squeeze()

    if occ is not None:
        r1 = occ[edge_indices[0, :]]
        r2 = occ[edge_indices[1, :]]
        # print(r1.shape, r2.shape)
        r = torch.max((r2 + eps) / (r1 + eps), (r1 + eps) / (r2 + eps))
        pvec = pvec / r
        # print(pvec.shape)
    pvec = torch.clamp(pvec, min=1e-5, max=1-1e-5)
    return pvec


class StrayAssigner(ABC):
    '''
    Abstract Class for orphan assigning functors.
    '''
    def __init__(self, X, Y, metric_fn : Callable = None):
        self.clustered = X
        self.d = metric_fn
        self.partial_labels = Y
        super().__init__()

    @abstractmethod
    def assign_orphans(self):
        pass

class NearestNeighborsAssigner(StrayAssigner):
    '''
    Assigns orphans to the k-nearest cluster using simple kNN Classifier.
    '''
    def __init__(self, X, Y, metric_fn : Callable = None, **kwargs):
        super(NearestNeighborsAssigner, self).__init__()
        self.k = kwargs.get('k', 10)
        self._neigh = KNeighborsClassifier(n_neighbors=10)
        self._neigh.fit(X)

    def assign_orphans(self, X, get_proba=False):
        pred = self._neigh.predict(X)
        self._pred = pred
        if get_proba:
            self._proba = self._neigh.predict_proba(orphans)
            self._max_proba = np.max(self._proba, axis=1)
        return pred


class ClusterGraphConstructor:
    '''
    Parametric Graph-SPICE clustering

    Parametric GDC includes a bilinear layer to predict edge weights,
    given pairs of node features.
    '''
    def __init__(self, constructor_cfg : dict,
                       graph_batch : GraphBatch = None,
                       graph_info : pd.DataFrame = None):

        # Input Data/Label conventions
        self.seg_col = constructor_cfg.get('seg_col', -1)
        self.cluster_col = constructor_cfg.get('cluster_col', 5)
        self.batch_col = constructor_cfg.get('batch_col', 3)
        self.training = False # Default mode is evaluation.

        # Initial Neighbor Graph Construction Mode
        mode = constructor_cfg.get('mode', 'knn')
        if mode == 'knn':
            self._init_graph = knn_graph
        elif mode == 'radius':
            self._init_graph = radius_graph
        else:
            raise ValueError('''Mode {} is not supported for initial
                graph construction!'''.format(mode))

        # Clustering Algorithm Parameters
        self.ths = constructor_cfg.get('edge_cut_threshold', 0.0) # Prob values 0-1
        # print("Edge Threshold Probability Score = ", self.ths)
        self.kwargs = constructor_cfg.get('cluster_kwargs', dict(k=5))

        # GraphBatch containing graphs per semantic class.
        if graph_batch is None:
            self._graph_batch = GraphBatch()
        else:
            assert graph_info is not None
            self._info = graph_info
            self._graph_batch = graph_batch
            self._num_total_nodes = self._graph_batch.x.shape[0]
            self._node_dim = self._graph_batch.x.shape[1]
            self._num_total_edges = self._graph_batch.edge_index.shape[1]

        self._edge_pred = None
        self._node_pred = None


    @staticmethod
    def get_edge_truth(edge_indices : torch.Tensor,
                       fragment_labels : torch.Tensor):
        '''
        Given edge indices and ground truth fragment labels,
        get true labels for binary edge classification.

        INPUTS:
            - edge_indices : 2 x E
            - labels : (N, ) Fragment label tensor

        RETURNS:
            - Edge labels : (N,) Tensor, where 0 indicate edges between
            different fragment voxels and 1 otherwise.
        '''
        u = fragment_labels[edge_indices[0, :]]
        v = fragment_labels[edge_indices[1, :]]
        return (u == v).long()


    def initialize_graph(self, res : dict,
                               labels: torch.Tensor):
        '''
        From GraphSPICE Embedder Output, initialize GraphBatch object
        with edge truth labels.

        Inputs:
            - res (dict): result dictionary output of GraphSPICE Embedder
            - labels ( N x F Tensor) :

        Transforms point cloud embeddings to collection of graphs
        (one per unique image id and segmentation class), and stores graph
        collection as attribute.
        '''
        features = res['hypergraph_features'][0]
        batch_indices = res['batch_indices'][0]
        coordinates = res['coordinates'][0]
        data_list = []

        # print(labels)

        graph_id = 0
        index = 0

        self._info = []

        for i, bidx in enumerate(torch.unique(batch_indices)):
            mask = batch_indices == bidx
            coords_batch = coordinates[mask]
            features_batch = features[mask]
            labels_batch = labels[mask].int()

            for c in torch.unique(labels_batch[:, self.seg_col]):
                class_mask = labels_batch[:, self.seg_col] == c
                coords_class = coords_batch[class_mask]
                features_class = features_batch[class_mask]

                edge_indices = self._init_graph(coords_class, **self.kwargs)
                data = GraphData(x=features_class,
                                 pos=coords_class,
                                 edge_index=edge_indices)
                graph_id_key = dict(Index=index,
                                    BatchID=int(bidx),
                                    SemanticID=int(c),
                                    GraphID=graph_id)
                graph_id += 1
                self._info.append(graph_id_key)

                if self.training:
                    frag_labels = labels_batch[class_mask][:, self.cluster_col]
                    truth = self.get_edge_truth(edge_indices, frag_labels)
                    data.edge_truth = truth
                data_list.append(data)
            index += 1

        self._info = pd.DataFrame(self._info)
        self.data_list = data_list
        self._graph_batch = self._graph_batch.from_data_list(data_list)
        self._num_total_nodes = self._graph_batch.x.shape[0]
        self._node_dim = self._graph_batch.x.shape[1]
        self._num_total_edges = self._graph_batch.edge_index.shape[1]


    def replace_state(self, graph_batch, info):
        self._graph_batch = graph_batch
        self._num_total_nodes = self._graph_batch.x.shape[0]
        self._node_dim = self._graph_batch.x.shape[1]
        self._num_total_edges = self._graph_batch.edge_index.shape[1]
        self._info = info


    def _set_edge_attributes(self, kernel_fn : Callable):
        '''
        Constructs edge attributes from node feature tensors, and saves
        edge attributes to current GraphBatch.
        '''
        if self._graph_batch is None:
            raise ValueError('The graph data has not been initialized yet!')
        elif isinstance(self._graph_batch.edge_attr, torch.Tensor):
            raise ValueError('Edge attributes are already set: {}'\
                .format(self._graph_batch.edge_attr))
        else:
            edge_attr = kernel_fn(
                self._graph_batch.x[self._graph_batch.edge_index[0, :]],
                self._graph_batch.x[self._graph_batch.edge_index[1, :]])
            w = edge_attr.squeeze()
            self._graph_batch.edge_attr = w
            self._graph_batch.add_edge_features(w, 'edge_attr')


    def get_batch_and_class(self, entry):
        df = self._info.query('GraphID == {}'.format(entry))
        assert df.shape[0] == 1
        batch_id = df['BatchID'].item()
        semantic_id = df['SemanticID'].item()
        return batch_id, semantic_id


    def get_entry(self, batch_id, semantic_id):
        df = self._info.query(
            'BatchID == {} and SemanticID == {}'.format(batch_id, semantic_id))
        assert df.shape[0] < 2
        if df.shape[0] == 0:
            raise ValueError('''Event ID: {} and Class Label: {} does not
                exist in current batch'''.format(batch_id, semantic_id))
            return None
        else:
            entry_num = df['GraphID'].item()
            return entry_num


    def get_graph(self, batch_id, semantic_id):
        '''
        Retrieve single graph from GraphBatch object by batch and semantic id.

        INPUTS:
            - event_id: Event ID (Index)
            - semantic_id: Semantic Class (0-4)

        RETURNS:
            - Subgraph corresponding to class [semantic_id] and event [event_id]
        '''
        entry_num = self.get_entry(batch_id, semantic_id)
        return self._graph_batch.get_example(entry_num)


    def fit_predict_one(self, entry,
                        gen_numpy_graph=False,
                        min_points=0,
                        cluster_all=True,
                        remainder_alg='knn') -> Tuple[np.ndarray, nx.Graph, GraphData]:
        '''
        Generate predicted fragment cluster labels for single subgraph.

        INPUTS:
            - entry number
            - gen_numpy_graph: whether to generate and output a networkx
            graph object with numpy converted graph attributes.
            - min_points: minimum voxel count required to assign
            unique cluster label during first pass.
            - cluster_all: if False, function will leave orphans as is
            with label -1.
            - remainder_alg: algorithm used to handle orphans

        Returns:
            - pred: predicted cluster labels.

        '''
        subgraph = self._graph_batch.get_example(entry)
        num_nodes = subgraph.num_nodes
        G = nx.Graph()
        G.add_nodes_from(np.arange(num_nodes))

        # Drop edges with low predicted probability score
        edges = subgraph.edge_index.T.cpu().numpy()
        edge_logits = subgraph.edge_attr.detach().cpu().numpy()
        edge_probs = expit(edge_logits)
        pos_edges = edges[edge_probs > self.ths]
        pos_probs = edge_probs[edge_probs > self.ths]
        pos_edges = [(e[0], e[1], w) for e, w in zip(pos_edges, pos_probs)]
        G.add_weighted_edges_from(pos_edges)
        pred = -np.ones(num_nodes, dtype=np.int32)
        for i, comp in enumerate(nx.connected_components(G)):
            if len(comp) < min_points:
                continue
            x = np.asarray(list(comp))
            pred[x] = i

        if gen_numpy_graph:
            G.x = subgraph.x.detach().cpu().numpy()
            G.edge_index = subgraph.edge_index.detach().cpu().numpy()
            G.edge_attr = subgraph.edge_attr.detach().cpu().numpy()
            G.pos = subgraph.pos.detach().cpu().numpy()

        new_labels, _ = unique_label(pred[pred >= 0])
        pred[pred >= 0] = new_labels

        return pred, G, subgraph


    def fit_predict(self, skip=[], **kwargs):
        '''
        Iterate over all subgraphs and assign predicted labels.
        '''
        skip = set(skip)
        num_graphs = self._graph_batch.num_graphs
        entry_list = [i for i in range(num_graphs) if i not in skip]
        node_pred = -np.ones(self._num_total_nodes, dtype=np.int32)

        pred_data_list = []

        for entry in entry_list:
            pred, G, subgraph = self.fit_predict_one(entry, **kwargs)
            batch_index = (self._graph_batch.batch.cpu().numpy() == entry)
            pred_data_list.append(GraphData(x=torch.Tensor(pred),
                                            pos=torch.Tensor(G.pos)))
            # node_pred[batch_index] = pred
        self._node_pred = GraphBatch.from_data_list(pred_data_list)
        # self._graph_batch.add_node_features(node_pred, 'node_pred',
        #                                     dtype=torch.long)


    def evaluate_nodes(self, cluster_label : np.ndarray,
                             metrics : List[ Callable ],
                             skip=[],
                             column_names=None):
        '''
        Evaluate accuracy metrics for node predictions using a list of
        scoring functions.

        INPUTS:
            - cluster_label : N x 6 Tensor, with pos, batch id,
            fragment_label, and segmentation label.
            - metrics : List of accuracy metric evaluation functions.
            - skip: list of graph ids to skip evaluation.

        Constructs a GraphBatch object containing true labels and stores it
        as an attribute to self.
        '''
        assert hasattr(self, '_node_pred')
        skip = set(skip)
        num_graphs = self._graph_batch.num_graphs
        entry_list = [i for i in range(num_graphs) if i not in skip]

        # Due to different voxel ordering convention, we need to create
        # a separate GraphBatch object for labels
        label_list = []
        if isinstance(cluster_label, torch.Tensor):
            batch_index = cluster_label[:, self.batch_col].int().cpu().numpy()
            segment_label = cluster_label[:, self.seg_col].int().cpu().numpy()
            fragment_label = cluster_label[:, self.cluster_col].int().cpu().numpy()
            label_pos = cluster_label[:, :3].int().cpu().numpy()
        else:
            batch_index = cluster_label[:, self.batch_col].astype(int)
            segment_label = cluster_label[:, self.seg_col].astype(int)
            fragment_label = cluster_label[:, self.cluster_col].astype(int)
            label_pos = cluster_label[:, :3]
        for bidx in np.unique(batch_index):
            batch_mask = batch_index == bidx
            labels_batch = cluster_label[batch_mask]
            slabels = segment_label[batch_mask]
            clabels = fragment_label[batch_mask]
            batch_pos = label_pos[batch_mask]
            for c in np.unique(slabels):
                gt = clabels[slabels == c]
                x = torch.Tensor(gt).to(dtype=torch.long)
                pos = batch_pos[slabels == c]
                d = GraphData(x=x, pos=pos)
                label_list.append(d)
        node_truth = GraphBatch.from_data_list(label_list)
        self._node_truth = node_truth

        add_columns = { f.__name__ : [] for f in metrics}
        if column_names is None:
            column_name_map = { f.__name__ : f.__name__ for f in metrics}
        else:
            column_name_map = { f.__name__ : name for f,name \
                                in zip(metrics, column_names)}

        for entry in entry_list:
            batch_id, semantic_id = self.get_batch_and_class(entry)
            subgraph = self.get_graph(batch_id, semantic_id)

            # Sort rows
            if isinstance(subgraph.pos, torch.Tensor):
                entry_pos = subgraph.pos.cpu().numpy()
            else:
                entry_pos = subgraph.pos
            # entry_perm = np.lexsort(entry_pos.T)
            # print(entry_pos[entry_perm])


            batch_index = (self._graph_batch.batch.cpu().numpy() == entry)
            labels = self._node_truth.get_example(entry).x
            temp = self._node_truth.get_example(entry).pos

            if isinstance(temp, torch.Tensor):
                label_pos = temp.cpu().numpy()
            else:
                label_pos = temp
            label_perm = np.lexsort(label_pos.T)
            # print(label_pos)
            # print(label_pos[label_perm])

            # assert False
            # print(self.node_pred.get_example(entry).pos)
            pred = self.node_pred.get_example(entry).x
            # print(pred.unique())
            # print(labels.unique())
            for f in metrics:
                score = f(pred, labels)
                # print(score)
                add_columns[f.__name__].append(score)

        self._info = self._info.assign(**add_columns)
        self._info.rename(columns=column_name_map, inplace=True)
        return self.info


    @property
    def node_pred(self):
        return self._node_pred

    @property
    def graph_batch(self):
        if self._graph_batch is None:
            raise('The GraphBatch data has not been initialized yet!')
        return self._graph_batch

    @property
    def info(self):
        '''
        Entry mapping (pd.DataFrame):

            - columns: ['Index', 'BatchID', 'SemanticID', 'GraphID']

        By querying on BatchID and SemanticID, for example, one obtains
        the graph id value (entry) used to access a single subgraph in
        self._graph_batch.
        '''
        return self._info


    def __call__(self, res : dict,
                       kernel_fn : Callable,
                       labels: torch.Tensor):
        '''
        Train time labels include cluster column (default: 5)
        and segmentation column (default: -1)
        Test time labels only include segment column (default: -1)
        '''
        self.initialize_graph(res, labels)
        self._set_edge_attributes(kernel_fn)
        return self._graph_batch


    def __repr__(self):
        out = '''
        ClusterGraphConstructor
        '''
        return out


# class CGCGNN(ClusterGraphConstructor):

#     def __init__(self, *args, **kwargs):
#         super(CGCGNN, self).__init__(*args, **kwargs)

#     def initialize_graph(self, res : dict,
#                                labels: torch.Tensor,
#                                gnn_model):
#         '''
#         From GraphSPICE Embedder Output, initialize GraphBatch object
#         with edge truth labels.

#         Inputs:
#             - res (dict): result dictionary output of GraphSPICE Embedder
#             - labels ( N x F Tensor) :

#         Transforms point cloud embeddings to collection of graphs
#         (one per unique image id and segmentation class), and stores graph
#         collection as attribute.
#         '''
#         features = res['hypergraph_features'][0]
#         batch_indices = res['batch_indices'][0]
#         coordinates = res['coordinates'][0]
#         data_list = []

#         # print(labels)

#         graph_id = 0

#         self._info = []

#         spatial_embeddings = res['spatial_embeddings'][0]
#         feature_embeddings = res['feature_embeddings'][0]
#         occupancy = res['occupancy'][0]
#         covariance = res['covariance'][0]

#         for i, bidx in enumerate(torch.unique(batch_indices)):
#             mask = batch_indices == bidx
#             coords_batch = coordinates[mask]
#             features_batch = features[mask]
#             labels_batch = labels[mask].int()

#             sp_batch = spatial_embeddings[mask]
#             ft_batch = feature_embeddings[mask]
#             occ_batch = occupancy[mask]
#             cov_batch = covariance[mask]

#             for c in torch.unique(labels_batch[:, self.seg_col]):
#                 class_mask = labels_batch[:, self.seg_col] == c
#                 coords_class = coords_batch[class_mask]
#                 features_class = features_batch[class_mask]

#                 sp_class = sp_batch[class_mask]
#                 ft_class = ft_batch[class_mask]
#                 cov_class = cov_batch[class_mask]
#                 occ_class = occ_batch[class_mask]

#                 edge_index = self._init_graph(coords_class, **self.kwargs)
#                 edge_weight = get_edge_weight(
#                     sp_class, ft_class, cov_class,
#                     edge_index=edge_index,
#                     occ=occ_class)

#                 print(edge_weight, edge_weight.shape)
#                 x = gnn_model(x=features_class,
#                               edge_index=edge_index,
#                               edge_weight=edge_weight)
#                 assert False
#                 data = GraphData(x=x,
#                                  pos=coords_class,
#                                  edge_index=edge_index,
#                                  edge_attr=edge_weight)
#                 graph_id_key = dict(Index=0,
#                                     BatchID=int(bidx),
#                                     SemanticID=int(c),
#                                     GraphID=graph_id)
#                 graph_id += 1
#                 self._info.append(graph_id_key)

#                 if self.training:
#                     frag_labels = labels_batch[class_mask][:, self.cluster_col]
#                     truth = self.get_edge_truth(edge_index, frag_labels)
#                     data.edge_truth = truth
#                 data_list.append(data)

#         self._info = pd.DataFrame(self._info)
#         self.data_list = data_list
#         self._graph_batch = self._graph_batch.from_data_list(data_list)
#         self._num_total_nodes = self._graph_batch.x.shape[0]
#         self._node_dim = self._graph_batch.x.shape[1]
#         self._num_total_edges = self._graph_batch.edge_index.shape[1]
