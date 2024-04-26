from typing import Union, Callable, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mlreco.utils.metrics import *

from sklearn.neighbors import (KNeighborsClassifier, 
                               RadiusNeighborsClassifier, 
                               kneighbors_graph)
import scipy.sparse as sp

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
from sklearn.cluster import DBSCAN

from mlreco import TensorBatch

import copy
# -------------------------- Helper Functions--------------------------------

def knn_sklearn(coords, k=5):
    if coords.shape[0] < k:
        n_neighbors = coords.shape[0] - 1
    else:
        n_neighbors = k
    if isinstance(coords, torch.Tensor):
        device = coords.device
        G = kneighbors_graph(coords.cpu().numpy(), n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return torch.Tensor(out).long().to(device=device)
    elif isinstance(coords, np.ndarray):
        G = kneighbors_graph(coords, n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return out


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
    def __init__(self, X, Y, metric_fn : Callable = None, **kwargs):
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
    def __init__(self, X, Y, **kwargs):
        '''
            X: Points to run Nearest-Neighbor Classifier (N x F)
            Y: Labels of Points (N, )
        '''
        super(NearestNeighborsAssigner, self).__init__(X, Y, **kwargs)
        self._neigh = KNeighborsClassifier(**kwargs)
        self._neigh.fit(X, Y)

    def assign_orphans(self, orphans, get_proba=False):
        pred = self._neigh.predict(orphans)
        self._pred = pred
        if get_proba:
            self._proba = self._neigh.predict_proba(orphans)
            self._max_proba = np.max(self._proba, axis=1)
        return pred


class RadiusNeighborsAssigner(StrayAssigner):
    '''
    Assigns orphans to the k-nearest cluster using simple kNN Classifier.
    '''
    def __init__(self, X, Y, **kwargs):
        '''
            X: Points to run Nearest-Neighbor Classifier (N x F)
            Y: Labels of Points (N, )
        '''
        super(RadiusNeighborsAssigner, self).__init__(X, Y, **kwargs)
        self._neigh = RadiusNeighborsClassifier(**kwargs)
        self._neigh.fit(X, Y)

    def assign_orphans(self, orphans, get_proba=False):
        pred = self._neigh.predict(orphans)
        self._pred = pred
        if get_proba:
            self._proba = self._neigh.predict_proba(orphans)
            self._max_proba = np.max(self._proba, axis=1)
        return pred
    
    
class RadiusNeighborsIterativeAssigner(StrayAssigner):
    
    def __init__(self, X, component, 
                 min_points=0, 
                 orphans_radius=1.9,
                 orphans_iterate=True,
                 orphans_cluster_all=True,
                 **kwargs):
        '''
            X: Points to run Nearest-Neighbor Classifier (N x F)
            Y: Labels of Points (N, )
        '''
        super(RadiusNeighborsIterativeAssigner, self).__init__(X, component, **kwargs)
        self._neigh = RadiusNeighborsClassifier(**kwargs)
        self._min_points = min_points
        labels, counts = np.unique(component, return_counts=True)
        self._pred = component
        self._labeled_mask = np.isin(component, labels[counts >= self._min_points])
        
        self._pred[~self._labeled_mask] = -1
        
        self.X = X
        self.Y = component
        
        self._orphans_iterate = kwargs.get('orphans_iterate', True)
        self._orphans_radius = kwargs.get('orphans_radius', 1.9)
        
    def assign_orphans(self):
        
        labeled_mask = copy.deepcopy(self._labeled_mask)
        num_orphans = 0
        if labeled_mask.all(): return self._pred
        if (~labeled_mask).all():
            self._pred = DBSCAN(eps=self._orphans_radius, min_samples=1).fit_predict(self.X)
            return self._pred
        while not np.all(labeled_mask) and num_orphans != np.sum(~labeled_mask):
            
            num_orphans = np.sum(~labeled_mask)
            
            X_valid, Y_valid = self.X[labeled_mask], self.Y[labeled_mask]
            self._neigh.fit(X_valid, Y_valid)
            
            orphan_labels = self._neigh.predict(self.X[~labeled_mask])
            valid_mask = orphan_labels > -1
            
            orphan_indices = np.where(~labeled_mask)[0][valid_mask]
            
            labeled_mask[orphan_indices] = True
            self._pred[orphan_indices] = orphan_labels[valid_mask]
            
            if not self._orphans_iterate: break
            
        return self._pred
    
    
class ConnectedComponents:
    def __init__(self, graph_state, orphan_mode='radius_neighbors'):
        self._graph_state = graph_state
        self.orphan_mode = orphan_mode

    def fit_predict_one(self, 
                        one_graph, 
                        edge_mode='edge_pred', 
                        offset=0, **kwargs):
        
        edge_index = one_graph['edge_index'].T
        assert edge_index.shape[0] == 2, 'Edge Index must be of shape (2, E)'
        labels = one_graph[edge_mode]

        connected_mask = (labels == 1).squeeze()
        connected_edges = edge_index[:, connected_mask]
        nodes = one_graph['node_coordinates']
        node_features = one_graph['node_features']
            
        if isinstance(edge_index, torch.Tensor):
            adj = to_scipy_sparse_matrix(connected_edges, num_nodes=len(nodes))
        else:
            edges, N = np.ones(len(connected_edges[0])), len(nodes)
            adj = sp.coo_matrix((edges, (connected_edges[0], connected_edges[1])), (N, N))

        num_components, component = sp.csgraph.connected_components(
            adj, connection='weak')
        
        component = component.astype(np.int64)
        
        if isinstance(nodes, torch.Tensor):
            assigner = RadiusNeighborsIterativeAssigner(nodes, component, **kwargs)
        else:
            assigner = RadiusNeighborsIterativeAssigner(nodes, component, **kwargs)
        node_pred = assigner.assign_orphans()

        if isinstance(node_features, torch.Tensor):
            device = node_features.device
            node_pred = torch.tensor(node_pred).long().to(device)

        node_pred[node_pred != -1] += offset
        return node_pred

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.connection})'
    
    def get_entry(self, batch_id, semantic_id):
        assert hasattr(self, '_graph_state')
        
        graph = {}
        node_clusts = self._graph_state['node_clusts'][batch_id][semantic_id]
        edge_clusts = self._graph_state['edge_clusts'][batch_id][semantic_id]
        
        for key, tensorbatch in self._graph_state.items():
            val = tensorbatch[batch_id]
            if isinstance(val, torch.Tensor):
                nclusts = torch.tensor(node_clusts, dtype=torch.long, device=val.device)
                eclusts = torch.tensor(edge_clusts, dtype=torch.long, device=val.device)
            else:
                nclusts = node_clusts
                eclusts = edge_clusts
            if key == 'node_clusts':
                continue
            if key == 'edge_clusts':
                continue
            if 'node' in key:
                graph[key] = val[nclusts]
            elif 'edge' in key:
                graph[key] = val[eclusts]
            else:
                graph[key] = val
        return graph
    
    @property
    def graph_state(self):
        return self._graph_state
    
    def forward(self,
                edge_mode='edge_pred', 
                **kwargs):
        assert hasattr(self, 'graph_state')
        
        node_coordinates = self._graph_state['node_coordinates']
        batch_size = node_coordinates.batch_size
        device = node_coordinates.device
        
        node_pred_all = []
        
        for batch_id in range(batch_size):
            offset = 0
            node_pred_batch = []
            node_clusts = self._graph_state['node_clusts'][batch_id]
            edge_clusts = self._graph_state['edge_clusts'][batch_id]
            assert len(node_clusts) == len(edge_clusts)
            
            if isinstance(node_coordinates.tensor, torch.Tensor):
                node_pred_batch = torch.ones(node_coordinates[batch_id].shape[0], dtype=torch.long, device=device)
            else:
                node_pred_batch = np.ones(node_coordinates[batch_id].shape[0], dtype=int)

            for semantic_id, nclusts in enumerate(node_clusts):
                one_graph = self.get_entry(batch_id, semantic_id)
                
                node_pred = self.fit_predict_one(one_graph,
                                                 edge_mode=edge_mode,
                                                 offset=offset,
                                                 **kwargs)
                nclusts_torch = torch.tensor(nclusts, dtype=torch.long, device=device)
                node_pred_batch[nclusts_torch] = node_pred
                
                offset = int(node_pred.max()) + 1
            
            node_pred_all.append(node_pred_batch)
        
        if isinstance(node_coordinates.tensor, torch.Tensor):
            node_pred_all = torch.hstack(node_pred_all)
        else:
            node_pred_all = np.hstack(node_pred_all)

        return TensorBatch(node_pred_all, counts=node_coordinates.counts)
            
    
class ConnectedComponentsDeprecated(BaseTransform):
    def __init__(self, connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.connection = connection

    def fit_predict_one(self, data: Data, 
                        edge_mode='edge_pred', 
                        offset=0, edge_threshold=0.1,
                        invert=True, **kwargs) -> Data:
        
        if edge_mode == 'edge_pred':
            edge_index = getattr(data, 'edge_index')
            connected_mask = (data.edge_pred == 1).squeeze()
            connected_edges = edge_index[:, connected_mask]
            edge_index = connected_edges
        elif edge_mode == 'edge_truth':
            edge_index = getattr(data, 'edge_index')
            connected_mask = (data.edge_label == 1).squeeze()
            connected_edges = edge_index[:, connected_mask]
            edge_index = connected_edges
        elif edge_mode == 'edge_prob':
            edge_index = getattr(data, 'edge_index')
            if invert:
                connected_mask = (data.edge_prob < edge_threshold).squeeze()
            else:
                connected_mask = (data.edge_prob >= edge_threshold).squeeze()
            connected_edges = edge_index[:, connected_mask]
            edge_index = connected_edges
        else:
            edge_index = getattr(data, edge_mode)
            
        assert edge_index.shape[0] == 2, 'Edge Index must be of shape (2, E)'
            
        if isinstance(edge_index, torch.Tensor):
            adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)
        else:
            edge_attr, N = np.ones(len(edge_index[0])), data.num_nodes
            adj = sp.coo_matrix((edge_attr, (edge_index[0], edge_index[1])), (N, N))

        num_components, component = sp.csgraph.connected_components(
            adj, connection=self.connection)
        
        component = component.astype(np.int64)
        
        if isinstance(data.pos, torch.Tensor):
            assigner = RadiusNeighborsIterativeAssigner(data.pos.cpu().numpy(), component, **kwargs)
        else:
            assigner = RadiusNeighborsIterativeAssigner(data.pos, component, **kwargs)
        node_pred = assigner.assign_orphans()

        if isinstance(data.x, torch.Tensor):
            device = data.x.device
            # data.node_pred = torch.tensor(node_pred).long().to(device) + offset
            data.node_pred = torch.tensor(node_pred).long().to(device)
        else:
            data.node_pred = node_pred

        data.node_pred[data.node_pred != -1] += offset

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.connection})'
    
    def forward(self, batch: Batch, edge_mode='edge_pred', **kwargs) -> Batch:
        
        x = 0
        
        data_list = batch.to_data_list()
        offset = 0
        current_batch = 0
        for graph_id, subgraph in enumerate(data_list):
            if current_batch < int(subgraph.graph_key[0]):
                # Next batch, set offset = 0
                offset = 0
                current_batch = int(subgraph.graph_key[0])
            
            self.fit_predict_one(subgraph, 
                                 edge_mode=edge_mode,
                                 offset=offset,
                                 **kwargs)

            offset = int(subgraph.node_pred.max()) + 1
            
        out = Batch.from_data_list(data_list)
        return out
