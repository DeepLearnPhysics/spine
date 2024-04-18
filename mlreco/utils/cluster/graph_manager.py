from typing import Union, Callable, Tuple, List, Dict

import numpy as np
import torch

from torch_cluster import knn_graph, radius_graph

from mlreco.utils.globals import *

from mlreco.utils.metrics import ARI, SBD, purity, efficiency
from mlreco.utils.data_structures import TensorBatch, IndexBatch, EdgeIndexBatch
from mlreco.utils.gnn.cluster import form_clusters_batch, form_clusters
from .helpers import ConnectedComponents, knn_sklearn
import sys


class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type predictions
    in GraphSPICE clustering.
    """
    
    def __init__(self, 
                 constructor_cfg : dict,
                 training=False):
        """Initialize the ClusterGraphConstructor.

        Parameters
        ----------
        graph_type : str, optional
            Type of graph to construct, by default 'knn'
        edge_cut_threshold : float, optional
            Thresholding value for edge prediction, by default 0.1
        training : bool, optional
            Whether we are in training mode or not, by default False

        Raises
        ------
        ValueError
            If the graph type is not supported.
        """
        self.constructor_cfg     = constructor_cfg
        self._graph_type         = constructor_cfg.get('mode', 'knn')
        self._edge_cut_threshold = constructor_cfg.get('edge_cut_threshold', 0.0) 
        self._graph_params       = constructor_cfg.get('cluster_kwargs', dict(k=5))
        self._skip_classes       = constructor_cfg.get('skip_classes', [4])
        self._min_points         = constructor_cfg.get('min_points', 0)
        self.target_col          = constructor_cfg.get('target_col', CLUST_COL)
        self.training = training
        
        # Graph Constructors
        if self._graph_type == 'knn':
            self._init_graph = knn_graph
        elif self._graph_type == 'radius':
            self._init_graph = radius_graph
        elif self._graph_type == 'knn_sklearn':
            self._init_graph = knn_sklearn
        else:
            msg = f"Graph type {self._graph_type} is not supported for "\
                "GraphSPICE initialzation!"
            raise ValueError(msg)
        
        # Clustering Algorithm Parameters
        self.ths = self._edge_cut_threshold # Prob values 0-1
        if self.training:
            self.ths = 0.0

        # Radius within which orphans get assigned to neighbor cluster
        self._orphans_radius      = constructor_cfg.get('orphans_radius', 1.9)
        self._orphans_iterate     = constructor_cfg.get('orphans_iterate', True)
        self._orphans_cluster_all = constructor_cfg.get('orphans_cluster_all', True)
        self.use_cluster_labels   = constructor_cfg.get('use_cluster_labels', True)
        
        
    def _make_graph(self, x, **kwargs):
        if isinstance(x, np.ndarray):
            nodes = torch.from_numpy(x).float()
            if torch.cuda.is_available():
                nodes = nodes.cuda()
        elif isinstance(x, torch.Tensor):
            nodes = x
        else:
            raise ValueError("Input must be a torch.Tensor or np.ndarray")
        edge_index = self._init_graph(x=nodes, **kwargs)
        return edge_index


    def get_edge_truth(self, edge_index, node_truth):
        """Compute the true edge labels based on the node labels.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor
        node_truth : torch.Tensor
            Node truth tensor

        Returns
        -------
        torch.Tensor
            Edge truth tensor
        """
        
        device = edge_index.device
        if isinstance(node_truth, np.ndarray):
            labels = torch.from_numpy(node_truth).to(device)
        else:
            labels = node_truth.to(device)
        
        edge_truth = torch.zeros(edge_index.shape[1], 
                                 device=edge_index.device, 
                                 dtype=torch.long)
        
        edge_truth[labels[edge_index[0]] == labels[edge_index[1]]] = 1
        return edge_truth
        
    
    def prune_labels(self, labels : TensorBatch, skip_classes=[]):
        if len(skip_classes) == 0:
            return labels
        else:
            if isinstance(labels.tensor, torch.Tensor):
                isin = torch.isin
                skip_list = torch.Tensor(skip_classes).to(labels.tensor.device)
            elif isinstance(labels.tensor, np.ndarray):
                isin = np.isin
                skip_list = skip_classes
            else:
                raise ValueError("Labels must be a torch.Tensor or np.ndarray")
            tensor = labels.tensor[~isin(labels.tensor[:, -1], skip_list)]
            out = TensorBatch(tensor, batch_size=labels.batch_size)
            return out
        
    def initialize(self, 
                   res : Dict[str, TensorBatch],
                   semantic_labels : TensorBatch,
                   kernel_fn : Callable,
                   invert=True,
                   cluster_labels : TensorBatch=None):
        
        coordinates = res['coordinates']
        features    = res['hypergraph_features']
        batch_size = coordinates.batch_size
        
        slabels = self.prune_labels(semantic_labels, self._skip_classes)
        if cluster_labels is not None:
            clabels = self.prune_labels(cluster_labels, self._skip_classes)
        else:
            clabels = None
        
        if isinstance(coordinates.tensor, torch.Tensor):
            hstack = torch.hstack
            vstack = torch.vstack
        elif isinstance(coordinates.tensor, np.ndarray):
            hstack = np.hstack
            vstack = np.vstack
        else:
            raise ValueError("Coordinates must be a torch.Tensor or np.ndarray")
        
        edge_index = []
        edge_shape = []
        edge_attr = []
        edge_truth = []
        edge_pred = []
        edge_prob = []
        edge_batch_ids = []
        
        node_clusts = form_clusters_batch(slabels.to_numpy(), min_size=2, column=SHAPE_COL)
        
        for i in range(coordinates.batch_size):
            coords_i = coordinates[i]
            features_i = features[i]
            labels_i = slabels[i]
            clusts_i = node_clusts[i]
            
            graph_i = self.build_single_graph(
                coords_i, features_i, clusts_i, labels_i, kernel_fn, invert)
            
            edge_index.append(graph_i['edge_index'])
            edge_shape.append(graph_i['edge_shape'])
            edge_attr.append(graph_i['edge_attr'])
            edge_truth.append(graph_i['edge_label'])
            edge_pred.append(graph_i['edge_pred'])
            edge_prob.append(graph_i['edge_prob'])
            
            edge_batch_ids.append(torch.full((graph_i['edge_index'].shape[1], 1), i, device=graph_i['edge_index'].device, dtype=torch.long))
            
        edge_index = hstack(edge_index)
        edge_shape = hstack(edge_shape)
        edge_attr = vstack(edge_attr)
        edge_truth = hstack(edge_truth)
        edge_pred = hstack(edge_pred)
        edge_prob = hstack(edge_prob)
        edge_batch_ids = vstack(edge_batch_ids)
        edge_batch_ids = TensorBatch(edge_batch_ids, batch_size=batch_size, batch_col=BATCH_COL)
        
        
        graph_state = {
            'gs_node_coordinates': coordinates,
            'gs_node_features': features,
            'gs_node_shape': slabels,
            'gs_edge_batch_ids': edge_batch_ids,
            'gs_edge_index': TensorBatch(edge_index.T, counts=edge_batch_ids.counts, batch_col=None),
            'gs_edge_shape': TensorBatch(edge_shape.view(-1, 1), counts=edge_batch_ids.counts, batch_col=None),
            'gs_edge_attr': TensorBatch(edge_attr.view(-1, 1), counts=edge_batch_ids.counts, batch_col=None),
            'gs_edge_label': TensorBatch(edge_truth.view(-1, 1), counts=edge_batch_ids.counts, batch_col=None),
            'gs_edge_pred': TensorBatch(edge_pred.view(-1, 1), counts=edge_batch_ids.counts, batch_col=None),
            'gs_edge_prob': TensorBatch(edge_prob.view(-1, 1), counts=edge_batch_ids.counts, batch_col=None),
        }
        
        edge_clusts = form_clusters_batch(graph_state['gs_edge_shape'].to_numpy(), min_size=0, column=0)
        
        graph_state['gs_node_clusts'] = node_clusts
        graph_state['gs_edge_clusts'] = edge_clusts
        
        if self.use_cluster_labels and clabels is not None:
            graph_state['gs_node_labels'] = TensorBatch(clabels.tensor[:, self.target_col], counts=coordinates.counts)
            
        self._graph_state = graph_state
        
        return graph_state
        
    def build_single_graph(self, 
                           coords_i : torch.Tensor,
                           features_i : torch.Tensor, 
                           clusts_i: torch.Tensor,
                           labels_i : torch.Tensor, 
                           kernel_fn : Callable, 
                           invert=True) -> Dict[str, torch.Tensor]:
        """Construct a neighbors graph for a single batch id and
        semantic class that will be used for connected components clustering.

        Parameters
        ----------
        coords_i : torch.Tensor
            N x 4 tensor of coordinates (batch_id, x, y, z)
        features_i : torch.Tensor
            Graph embedding features to be used for edge prediction
        labels_i : torch.Tensor
            N x 5 Label tensor, where the last column is the semantic class
        kernel_fn : Callable
            Bilinear kernel function that computes edge score values.
        invert : bool, optional
            Whether to treat connected edges to have label 0 (True) or 1 (False).

        Returns
        -------
        Dict[str, torch.Tensor]
            _description_
        """
        
        edge_index_batch = []
        edge_shape_batch = []
        edge_truth_batch = []
        
        # Loop over each semantic class
        for shape_id in torch.unique(labels_i[:, SHAPE_COL]):
            shape_mask = labels_i[:, SHAPE_COL] == shape_id
            coords_slice = coords_i[shape_mask]
            # Build edges
            e = self._make_graph(x=coords_slice, 
                                 **self._graph_params)
            edge_index_batch.append(e)
            edge_shape_batch.append(
                torch.full((e.shape[1], ), 
                            shape_id, 
                            device=e.device, 
                            dtype=torch.long))
            # Only during training: compute edge truth labels
            if self.use_cluster_labels:
                node_truth = labels_i[shape_mask, self.target_col]
                e_truth = self.get_edge_truth(e, node_truth)
                edge_truth_batch.append(e_truth)
            
        edge_index_batch = torch.hstack(edge_index_batch)
        edge_shape_batch = torch.hstack(edge_shape_batch)
        
        # Compute true edges, if labels are provided
        if len(edge_truth_batch) > 0 and self.use_cluster_labels:
            edge_truth_batch = torch.hstack(edge_truth_batch)
        else:
            edge_truth_batch = torch.Tensor([], 
                                            device=coords_i.device, 
                                            dtype=torch.long)
        
        edge_attr_batch = kernel_fn(
            features_i[edge_index_batch[0, :]],
            features_i[edge_index_batch[1, :]])
        
        # Compute predicted edges
        edge_pred_batch, edge_prob_batch = self.predict_edges(edge_attr_batch, 
                                                              invert=invert)
        
        out = {
            'edge_index': edge_index_batch,
            'edge_shape': edge_shape_batch,
            'edge_attr': edge_attr_batch,
            'edge_label': edge_truth_batch,
            'edge_pred': edge_pred_batch,
            'edge_prob': edge_prob_batch
        }
        
        return out
    
    def predict_edges(self, edge_attr_batch, invert=True):
        """Predict edges based on edge attributes.

        Parameters
        ----------
        edge_attr_batch : torch.Tensor
            Edge attributes
        invert : bool, optional
            Whether to invert the edge prediction, by default True

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted edges and edge probabilities
            Each tensor has shape (N, )
        """
        device      = edge_attr_batch.device
        edge_prob   = torch.sigmoid(edge_attr_batch).to(device)
        edge_pred   = torch.zeros_like(edge_prob).long().to(device)
        
        if invert:
            # If invert=True, model is trained to predict disconnected edges
            # as positives. Hence connected edges are those with scores less
            # than the thresholding value. (Higher the value, the more likely
            # it is to be disconnected). 
            mask = (edge_prob < self.ths).squeeze()
            edge_pred[mask] = 1
        else:
            # When trained to predict connected edges. 
            mask = (edge_prob >= self.ths).squeeze()
            edge_pred[mask] = 1
            
        return edge_pred.squeeze(), edge_prob.squeeze()
    
    def get_entry(self, batch_id, semantic_id):
        assert hasattr(self, '_graph_state')
        
        graph = {}
        node_clusts = self._graph_state['gs_node_clusts'][batch_id][semantic_id]
        edge_clusts = self._graph_state['gs_edge_clusts'][batch_id][semantic_id]
        
        for key, tensorbatch in self._graph_state.items():
            val = tensorbatch[batch_id]
            if isinstance(val, torch.Tensor):
                nclusts = torch.tensor(node_clusts, dtype=torch.long, device=val.device)
                eclusts = torch.tensor(edge_clusts, dtype=torch.long, device=val.device)
            else:
                nclusts = node_clusts
                eclusts = edge_clusts
            if key == 'gs_node_clusts':
                continue
            if key == 'gs_edge_clusts':
                continue
            if 'gs_node' in key:
                graph[key] = val[nclusts]
            elif 'gs_edge' in key:
                graph[key] = val[eclusts]
            else:
                graph[key] = val
        return graph
        
    def save_state(self):
        return self._graph_state
    
    def load_state(self, res):
        graph_state = {}
        for key, val in res.items():
            if key.startswith('gs_'):
                graph_state[key] = val
        self._graph_state = graph_state
        
    def fit_predict(self,
                    edge_mode='edge_pred',
                    min_points=0):
        '''Perform connected components clustering on the graph.

        Parameters
        ----------
        edge_mode : str, optional
            _description_, by default 'edge_pred'
        min_points : int, optional
            _description_, by default 0
        edge_threshold : float, optional
            _description_, by default 0.1
        invert : bool, optional
            _description_, by default True
        '''
        with torch.no_grad():
            cc_predictor = ConnectedComponents(self._graph_state)
            node_pred = cc_predictor.forward(edge_mode, 
                                            min_points=min_points, 
                                            orphans_radius=self._orphans_radius, 
                                            orphans_iterate=self._orphans_iterate,
                                            orphans_cluster_all=self._orphans_cluster_all,
                                            outlier_label=int(-1))
            
            self._graph_state['gs_node_pred'] = node_pred
            
            return node_pred
    
    def evaluate(self):
        
        assert hasattr(self, 'graph_state')
        
        out = {
            'ari': [],
            'purity': [],
            'efficiency': []
        }
        
        for c in SHAPE_LABELS:
            if c < 4:
                out[f'ari_{c}'] = []
                out[f'purity_{c}'] = []
                out[f'efficiency_{c}'] = []
                
        with torch.no_grad():
        
            node_coordinates = self._graph_state['gs_node_coordinates']
            batch_size = node_coordinates.batch_size
            device = node_coordinates.device
            
            for batch_id in range(batch_size):

                node_clusts = self._graph_state['gs_node_clusts'][batch_id]
                edge_clusts = self._graph_state['gs_edge_clusts'][batch_id]
                
                for semantic_id, nclusts in enumerate(node_clusts):
                    one_graph = self.get_entry(batch_id, semantic_id)
                    node_pred = one_graph['gs_node_pred']
                    node_truth = one_graph['gs_node_labels']
                    
                    if len(node_truth.shape) == 0 or len(node_pred.shape) == 0:
                        continue
                    assert node_truth.shape[0] == node_pred.shape[0]
                    
                    ari = ARI(node_pred, node_truth)
                    pur = purity(node_pred, node_truth)
                    eff = efficiency(node_pred, node_truth)
                    
                    out['ari'].append(ari)
                    out['purity'].append(pur)
                    out['efficiency'].append(eff)
                    
                    out[f'ari_{semantic_id}'].append(ari)
                    out[f'purity_{semantic_id}'].append(pur)
                    out[f'efficiency_{semantic_id}'].append(eff)
                    
        for key, l in out.items():
            if len(l) == 0:
                out[key] = -sys.maxsize
            else:
                # Batch-averaged classwise accuracies
                out[key] = sum(l) / len(l)
            
        return out
    
    # def __call__(self, res: dict,
    #              kernel_fn: Callable,
    #              labels: Union[torch.Tensor, list],
    #              cluster_labels=None,
    #              state_dict=None,
    #              invert=True):
    #     if state_dict is None:
    #         self.initialize(res, labels, kernel_fn, invert=invert, cluster_labels=cluster_labels)
    #     else:
    #         self.load_state(state_dict)
    #     return self._graph_state
    
    def __repr__(self):
        msg = f"""
        ClusterGraphConstructor(
            constructor_cfg={self.constructor_cfg},
            training={self.training},
            cc_predictor={self._cc_predictor.__repr__()},
            data={self._data.__repr__()})
        """
        return msg