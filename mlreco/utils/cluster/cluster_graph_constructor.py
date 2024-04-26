from typing import Union, Callable, Tuple, List

# Silencing this warning about unclassified (-1) voxels
# /usr/local/lib/python3.8/dist-packages/sklearn/neighbors/_classification.py:601: 
# UserWarning: Outlier label -1 is not in training classes. 
# All class probabilities of outliers will be assigned with 0.
# warnings.warn('Outlier label {} is not in training '
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

import sys

import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph

from mlreco.utils.metrics import *
from mlreco.utils.globals import *
from torch_geometric.data import Data, Batch

from mlreco import TensorBatch, IndexBatch, EdgeIndexBatch
from mlreco.utils.metrics import ARI, SBD, purity, efficiency

from mlreco import TensorBatch, IndexBatch, EdgeIndexBatch
from mlreco.utils.gnn.cluster import form_clusters_batch, form_clusters
from .helpers import ConnectedComponentsDeprecated, knn_sklearn
import sys

class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type predictions
    in GraphSPICE clustering.
    """
    ATTR_NAMES = [
        'edge_index',
        'edge_attr',
        'edge_label',
        'edge_prob',
        'edge_pred',
        'edge_batch',
        'edge_image_id',
        'x',
        'pos',
        'node_pred',
        'node_truth',
        'image_id',
        'voxel_id',
        'semantic_id'
    ]
    
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
        self._skip_classes       = constructor_cfg.get('skip_classes', [])
        self._min_points         = constructor_cfg.get('min_points', 0)
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
        
        self._data = None
        self._key_to_graph = {}
        self._graph_to_key = {}
        self._cc_predictor = ConnectedComponentsDeprecated()
        
    
    def clear_data(self):
        """Clear the data stored in the constructor.
        """
        self._data = None
        self._key_to_graph = {}
        self._graph_to_key = {}
        
        
    def initialize_graph(self, res : dict,
                               labels: Union[torch.Tensor, list],
                               kernel_fn: Callable,
                               unwrapped=False,
                               invert=False):
        """Initialize the graph for a given batch.

        Parameters
        ----------
        res : dict
            Results dictionary from the network.
        labels : Union[torch.Tensor, list]
            Labels for the batch.
        kernel_fn : Callable
            Kernel function for computing edge weights.
        unwrapped : bool, optional
            Indicates whether the results are unwrapped or not, by default False
        invert : bool, optional
            Indicates whether the model is trained to predict disconnected edges.

        Returns
        -------
        None
            Operates in place.
        """
        self.clear_data()
        
        if unwrapped:
            return self._initialize_graph_unwrapped(res, labels, kernel_fn,
                                                    invert=invert)

        # features = res['hypergraph_features'].tensor
        # batch_indices = res['coordinates'].tensor[:, BATCH_COL].int()
        # coordinates = res['coordinates'].tensor[:, COORD_COLS]
        
        coordinates = res['coordinates']
        features = res['hypergraph_features']
        
        voxel_indices = TensorBatch(torch.arange(coordinates.tensor.shape[0]).long().to(coordinates.device), 
                                    counts=coordinates.counts,
                                    batch_col=None)

        data_list = []
        graph_id  = 0

        for i in range(coordinates.batch_size):
            coords_batch = coordinates[i]
            features_batch = features[i]
            labels_batch = labels[i].int()
            voxel_indices_batch = voxel_indices[i]

            for c in torch.unique(labels_batch[:, SHAPE_COL]):
                data = self._construct_graph_data(int(i),
                                    int(c),
                                    graph_id,
                                    coords_batch,
                                    features_batch,
                                    labels_batch,
                                    voxel_indices_batch,
                                    kernel_fn=kernel_fn,
                                    invert=invert)
                data_list.append(data)
                graph_id += 1
        self._data = Batch.from_data_list(data_list)

        
    def initialize_graph_unwrapped(self, res, labels, kernel_fn,
                                   invert=False):
        """Same as initialize_graph, but for unwrapped results.
        
        TODO: Fix unwrapped behavior for DDP
        """
        raise NotImplementedError("Unwrapped behavior not yet implemented for DDP!")
        self.clear_data()
        
        features    = res['hypergraph_features']
        batch_size  = len(res['coordinates'])
        coordinates = res['coordinates']
        
        assert batch_size == len(features)
        assert batch_size == len(coordinates)
        
        data_list = []
        graph_id  = 0
        
        for bidx in range(batch_size):
            coords_batch   = coordinates[bidx][:, COORD_COLS]
            features_batch = features[bidx]
            labels_batch   = labels[bidx]
            voxel_indices_batch =  torch.arange(
                coords_batch.shape[0]).int().to(coords_batch.device)
            
            for c in torch.unique(labels_batch[:, SHAPE_COL]):
                data = self._construct_graph_data(int(bidx),
                                                  int(c),
                                                  graph_id,
                                                  coords_batch,
                                                  features_batch,
                                                  labels_batch,
                                                  voxel_indices_batch,
                                                  kernel_fn=kernel_fn,
                                                  invert=invert)
                data_list.append(data)
                graph_id += 1
        self._data = Batch.from_data_list(data_list)
        

    def _construct_graph_data(self, 
                              batch_id,
                              semantic_type,
                              graph_id,
                              coords_batch, 
                              features_batch, 
                              labels_batch,
                              voxel_indices_batch,
                              kernel_fn,
                              invert=False,
                              build_graph=True,
                              edge_index=None) -> Data:
        """Construct a single graph for a given batch, semantic type pair.
        """
        
        class_mask = labels_batch[:, SHAPE_COL] == semantic_type
        coords_class = coords_batch[class_mask]
        features_class = features_batch[class_mask]
        voxels_class = voxel_indices_batch[class_mask]

        if build_graph:
            edge_index = self._init_graph(coords_class, **self._graph_params)
        else:
            assert edge_index.shape[0] == 2
        edge_attr  = kernel_fn(
            features_class[edge_index[0, :]],
            features_class[edge_index[1, :]])
        
        image_id = int(batch_id) * torch.ones(features_class.shape[0],
                                            dtype=torch.long, 
                                            device=features_class.device)
        voxel_id = voxels_class
        semantic_id = int(semantic_type) * torch.ones(features_class.shape[0],
                                                           dtype=torch.long,
                                                           device=features_class.device)
        
        data = Data(x=features_class,
                    pos=coords_class,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    image_id=image_id,
                    voxel_id=voxel_id,
                    semantic_id=semantic_id)
        
        # Mappings from GraphID to (BatchID, SemanticID)
        data.graph_id = int(graph_id)
        data.graph_key = (int(batch_id), int(semantic_type))
        self._key_to_graph[(int(batch_id), int(semantic_type))] = graph_id
        self._graph_to_key[graph_id] = (int(batch_id), int(semantic_type))
        self._predict_edges(data, invert=invert)

        if self.use_cluster_labels:
            frag_labels = labels_batch[class_mask][:, CLUST_COL]
            truth = self.get_edge_truth(edge_index, frag_labels)
            data.node_truth = frag_labels.view(-1, 1).long()
            data.edge_label = truth.view(-1, 1)
        return data

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
        
    def get_graph_at(self, batch_id, semantic_id):
        """Retrieve the graph at a given batch, semantic type pair.

        Parameters
        ----------
        batch_id : int
            Batch ID.
        semantic_id : int
            Semantic type ID.

        Returns
        -------
        subgraph: torch_geometric.data.Data
            One subgraph corresponding to a unique (batch, semantic type) pair.
        """
        graph_id = self._key_to_graph((batch_id, semantic_id))
        return self._data.get_example(graph_id)
    
    def get_example(self, graph_id, verbose=False):
        if verbose:
            key = self._graph_to_key[graph_id]
            print(f"Graph at BatchID={key[0]} and SemanticID={key[1]}")
        return self._data.get_example(graph_id)
    
    def __getitem__(self, graph_id):
        return self.get_example(graph_id)
    
    def _predict_edges(self, data, invert=False):
        
        device = data.edge_attr.device
        edge_logits = data.edge_attr
        edge_probs  = torch.sigmoid(edge_logits).to(device)
        edge_pred   = torch.zeros_like(edge_probs).long().to(device)
        
        if invert:
            # If invert=True, model is trained to predict disconnected edges
            # as positives. Hence connected edges are those with scores less
            # than the thresholding value. (Higher the value, the more likely
            # it is to be disconnected). 
            mask = (edge_probs < self.ths).squeeze()
            edge_pred[mask] = 1
        else:
            # When trained to predict connected edges. 
            mask = (edge_probs >= self.ths).squeeze()
            edge_pred[mask] = 1
            
        data.edge_pred = edge_pred
        data.edge_prob = edge_probs
    
    
    def save_state(self, unwrapped=False):
        
        state_dict = defaultdict(list)

        batch = self._data.batch
        edge_index = self._data.edge_index
        image_id = self._data.image_id
        data_list = self._data.to_data_list()
        for graph_id, subgraph in enumerate(data_list):
            for attr_name in self.ATTR_NAMES:
                if hasattr(subgraph, attr_name):
                    if attr_name == 'edge_index':
                        state_dict[attr_name].append(subgraph.edge_index.T)
                    else:
                        state_dict[attr_name].append(getattr(subgraph, attr_name))
            state_dict['graph_id'].append(int(subgraph.graph_id))
            state_dict['graph_key'].append(subgraph.graph_key)
    
        if unwrapped:
            return state_dict
        else:
            state_dict_wrapped = {}

            for key, val in state_dict.items():
                if isinstance(val[0], torch.Tensor):
                    state_dict_wrapped[key] = torch.cat(val, dim=0)
                else:
                    state_dict_wrapped[key] = np.array(val).astype(int)
                    
            batch_size = int(image_id.max()) + 1
            
            # Convert to TensorBatch
            out = {}
            
            # Reference Tensor (Nodes)
            out['pos'] = TensorBatch(state_dict_wrapped['pos'], 
                                     batch_size=batch_size, 
                                     batch_col=BATCH_COL)
            # Reference Tensor (Edges)
            edge_image_id = image_id[edge_index[0, :]].view(-1, 1)
            edge_index_with_batch = torch.cat([edge_image_id, 
                                               state_dict_wrapped['edge_index']], dim=1)
            out['edge_index'] = TensorBatch(edge_index_with_batch, 
                                               batch_size=batch_size)
            
            # Clusts
            batch_np = TensorBatch(batch.cpu().numpy().reshape(-1, 1), counts=out['pos'].counts)
            out['node_clusts'] = form_clusters_batch(batch_np, column=0, min_size=2)
            
            edge_batch_np = TensorBatch(batch[edge_index[0, :]].cpu().numpy().reshape(-1, 1),
                                        counts=out['edge_index'].counts)
            
            out['edge_clusts'] = form_clusters_batch(edge_batch_np, column=0, batch_size=batch_size)
            
            # Graphs
            graph_key = torch.tensor(state_dict_wrapped['graph_key'])
            graph_key = torch.cat([graph_key[:, 0].view(-1, 1), graph_key], dim=1)
            out['graph_key'] = TensorBatch(graph_key, batch_size=batch_size, batch_col=BATCH_COL)
            
            # Everything else
            for key, val in state_dict_wrapped.items():
                if key not in out:
                    if len(val.shape) < 2:
                        if type(val) is np.ndarray:
                            val = val.reshape(-1, 1)
                        elif type(val) is torch.Tensor:
                            val = val.view(-1, 1)
                        else:
                            raise ValueError(f"Unsupported type for {key}!")
                    if val.shape[0] == out['pos'].tensor.shape[0]:
                        out[key] = TensorBatch(val, counts=out['pos'].counts, batch_col=None)
                    elif val.shape[0] == out['edge_index'].tensor.shape[0]:
                        out[key] = TensorBatch(val, counts=out['edge_index'].counts, batch_col=None)
                    elif val.shape[0] == out['graph_key'].tensor.shape[0]:
                        out[key] = TensorBatch(val, counts=out['graph_key'].counts, batch_col=None)
                    else:
                        raise ValueError(f"Shape mismatch for {key}!")
            
            return out

    
    def _load_state_wrapped(self, state_dict):
        self.clear_data()
        data_list = []
        optionals = ['node_pred', 'node_truth', 
                     'edge_label', 'edge_pred']
        
        num_graphs = len(state_dict['graph_id'].tensor)
        
        node_clusts = state_dict['node_clusts']
        edge_clusts = state_dict['edge_clusts']

        assert len(node_clusts) == len(edge_clusts)

        batch_size = len(node_clusts)
        
        for i in range(batch_size):
            nclusts = node_clusts[i]
            eclusts = edge_clusts[i]
            if len(nclusts) != len(eclusts):
                raise AssertionError(f"Number of node and edge clusters do not match! {len(nclusts)} != {len(eclusts)}")
            graph_ids = state_dict['graph_id'][i]
            for j, nodes in enumerate(nclusts):
                subgraph = Data(x=state_dict['x'][i][nodes],
                                pos=state_dict['pos'][i][nodes, 1:],
                                edge_index=state_dict['edge_index'][i].T[1:, eclusts[j]],
                                edge_attr=state_dict['edge_attr'][i][eclusts[j]],
                                edge_pred=state_dict['edge_pred'][i][eclusts[j]],
                                edge_prob=state_dict['edge_prob'][i][eclusts[j]],
                                image_id=state_dict['image_id'][i][nodes],
                                voxel_id=state_dict['voxel_id'][i][nodes],
                                semantic_id=state_dict['semantic_id'][i][nodes])
                for name in optionals:
                    if name in state_dict and name.startswith('node'):
                        setattr(subgraph, name, state_dict[name][i][nodes])
                    if name in state_dict and name.startswith('edge'):
                        setattr(subgraph, name, state_dict[name][i][eclusts[j]])
                subgraph.graph_id = int(graph_ids[j])
                subgraph.graph_key = (int(state_dict['graph_key'][i][j][1]), 
                                      int(state_dict['graph_key'][i][j][2]))
                data_list.append(subgraph)
                
        self._data = Batch.from_data_list(data_list)

    
    def load_state(self, state_dict, unwrapped=False):
        
        if not unwrapped:
            self._load_state_wrapped(state_dict)
        else:
            raise NotImplementedError("Unwrapped state loading not yet implemented!")
            self.clear_data()
            data_list = []
            optionals  = ['node_pred', 'node_truth', 
                        'edge_label', 'edge_pred']
            num_graphs = len(state_dict['x'])
            for i in range(num_graphs):
                subgraph = Data(x=state_dict['x'][i],
                                pos=state_dict['pos'][i],
                                edge_index=state_dict['edge_index'][i].T,
                                edge_attr=state_dict['edge_attr'][i],
                                edge_pred=state_dict['edge_pred'][i],
                                edge_prob=state_dict['edge_prob'][i],
                                image_id=state_dict['image_id'][i],
                                voxel_id=state_dict['voxel_id'][i],
                                semantic_id=state_dict['semantic_id'][i])
                for name in optionals:
                    if name in state_dict:
                        setattr(subgraph, name, state_dict[name][i])
                        
                subgraph.graph_id = int(state_dict['graph_id'][i])
                subgraph.graph_key = tuple(state_dict['graph_key'][i])
                self._key_to_graph[state_dict['graph_key'][i]] = state_dict['graph_id'][i]
                self._graph_to_key[state_dict['graph_id'][i]] = state_dict['graph_key'][i]
                data_list.append(subgraph)
            
            self._data = Batch.from_data_list(data_list)
            

    def fit_predict(self, 
                    edge_mode='edge_pred', 
                    min_points=None,
                    edge_threshold=0.1,
                    invert=True):
        """Run GraphSPICE clustering on all graphs in the batch.

        Parameters
        ----------
        skip : list, optional
            Semantic types to skip, by default [-1, 4]
        edge_mode : str, optional
            Edge mode to use for clustering, by default 'edge_pred'

        Returns
        -------
        node_pred : torch.Tensor
            Predicted cluster labels for each voxel.
        """
        if min_points is None:
            min_points = self._min_points
        graphs = self._cc_predictor.forward(self._data, edge_mode=edge_mode,
                                            min_points=min_points,
                                            orphans_radius=self._orphans_radius,
                                            orphans_iterate=self._orphans_iterate,
                                            orphans_cluster_all=self._orphans_cluster_all,
                                            outlier_label=int(-1),
                                            edge_threshold=edge_threshold,
                                            invert=invert)
        self._data = graphs

        return graphs
    
    
    def evaluate(self):
        
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
            
            for data in self._data.to_data_list():
                
                batch_id, semantic_type = data.graph_key

                node_truth = data.node_truth.detach().cpu().numpy().squeeze()
                node_pred = data.node_pred.detach().cpu().numpy().squeeze()
            
                if len(node_truth.shape) == 0 or len(node_pred.shape) == 0:
                    continue
                assert node_truth.shape[0] == node_pred.shape[0]
                
                ari = ARI(node_pred, node_truth)
                pur = purity(node_pred, node_truth)
                eff = efficiency(node_pred, node_truth)
                
                out['ari'].append(ari)
                out['purity'].append(pur)
                out['efficiency'].append(eff)
                
                out[f'ari_{semantic_type}'].append(ari)
                out[f'purity_{semantic_type}'].append(pur)
                out[f'efficiency_{semantic_type}'].append(eff)
                
        for key, l in out.items():
            if len(l) == 0:
                out[key] = -sys.maxsize
            else:
                # Batch-averaged classwise accuracies
                out[key] = sum(l) / len(l)
            
        return out
        
    
    def __call__(self, res: dict,
                 kernel_fn: Callable,
                 labels: Union[torch.Tensor, list],
                 unwrapped=False,
                 state_dict=None,
                 invert=False):
        if state_dict is None:
            self.initialize_graph(res, labels, kernel_fn, unwrapped=unwrapped, invert=invert)
            # node_pred = self.fit_predict(self._skip_classes)
        else:
            self.load_state(state_dict)
            # if 'node_pred' not in state_dict:
            #     node_pred = self.fit_predict(self._skip_classes)
        return self._data
    
    def __repr__(self):
        msg = f"""
        ClusterGraphConstructor(
            constructor_cfg={self.constructor_cfg},
            training={self.training},
            cc_predictor={self._cc_predictor.__repr__()},
            data={self._data.__repr__()})
        """
        return msg


#-------------------HELPER FUNCTIONS----------------------

def num_true_clusters(pred, truth):
    return len(np.unique(truth))

def num_pred_clusters(pred, truth):
    return len(np.unique(pred))

def num_small_clusters(pred, truth, threshold=5):
    val, cnts = np.unique(pred, return_counts=True)
    return np.count_nonzero(cnts < threshold)

def modified_ARI(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return ARI(pred[mask & mask2], truth[mask & mask2])

def modified_purity(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return purity(pred[mask & mask2], truth[mask & mask2])

def modified_efficiency(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return efficiency(pred[mask & mask2], truth[mask & mask2])
