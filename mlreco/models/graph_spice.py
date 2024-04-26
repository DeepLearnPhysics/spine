import torch
import numpy as np
import MinkowskiEngine as ME

from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.layers.cluster_cnn import gs_kernel_factory, spice_loss_factory

from mlreco.models.layers.cluster_cnn.graph_spice_embedder import GraphSPICEEmbedder

from mlreco import TensorBatch
from mlreco.utils.cluster.graph_manager import ClusterGraphConstructor
from mlreco.utils.unwrap import Unwrapper
from mlreco.utils.globals import *


class GraphSPICE(nn.Module):
    '''
    Neighbor-graph embedding based particle clustering.

    GraphSPICE has two components:

    1. Voxel Embedder: UNet-type CNN architecture used for feature
    extraction and feature embeddings.

    2. Edge Probability Kernel function: A kernel function (any callable
    that takes two node attribute vectors to give a edge proability score).

    Prediction is done in two steps:

    1. A neighbor graph (ex. KNN, Radius) is constructed to compute
    edge probabilities between neighboring edges.

    2. Edges with low probability scores are dropped.
    
    3. The voxels are clustered by counting connected components.

    Configuration
    -------------
    skip_classes: list, default [2, 3, 4]
        semantic labels for which to skip voxel clustering
        (ex. Michel, Delta, and Low Es rarely require neural network clustering)
    dimension: int, default 3
        Spatial dimension (2 or 3).
    min_points: int, default 0
        If a value > 0 is specified, this will enable the orphans assignment for
        any predicted cluster with voxel count < min_points.

        .. warning::

            ``min_points`` is set to 0 at training time.

    node_dim: int
    use_raw_features: bool
    constructor_cfg: dict
        Configuration for ClusterGraphConstructor instance. A typical configuration:

        .. code-block:: yaml

              constructor_cfg:
                mode: 'knn'
                seg_col: -1
                cluster_col: 5
                edge_mode: 'attributes'
                hyper_dimension: 22
                edge_cut_threshold: 0.1

        .. warning::

            ``edge_cut_threshold`` is set to 0. at training time.
            At inference time you want to set it to a value > 0.
            As a rule of thumb, 0.1 is a good place to start.
            Its exact value can be optimized.

    embedder_cfg: dict
        A typical configuration would look like:

        .. code-block:: yaml

              embedder_cfg:
                graph_spice_embedder:
                  segmentationLayer: False
                  feature_embedding_dim: 16
                  spatial_embedding_dim: 3
                  num_classes: 5
                  occupancy_mode: 'softplus'
                  covariance_mode: 'softplus'
                uresnet:
                  filters: 32
                  input_kernel: 5
                  depth: 5
                  reps: 2
                  spatial_size: 768
                  num_input: 4 # 1 feature + 3 normalized coords
                  allow_bias: False
                  activation:
                    name: lrelu
                    args:
                      negative_slope: 0.33
                  norm_layer:
                    name: batch_norm
                    args:
                      eps: 0.0001
                      momentum: 0.01

    kernel_cfg: dict
        A typical configuration:

        .. code-block:: yaml

              kernel_cfg:
                name: 'bilinear'
                num_features: 32

    .. warning::

        Train time and test time configurations are slightly different for GraphSpice.

    Output
    ------
    graph:
    graph_info:
    coordinates:
    hypergraph_features:

    See Also
    --------
    GraphSPICELoss
    '''

    MODULES = ['constructor_cfg', 'embedder_cfg', 'kernel_cfg', 'gspice_fragment_manager']

    def __init__(self, graph_spice, graph_spice_loss=None):
        """Initialize the S3C (Supervised Conn. Components Clustering) Model

        Parameters
        ----------
        graph_spice : dict
            GraphSPICE configuration dictionary
        name : str, optional
            _description_, by default 'graph_spice'
        """
        super(GraphSPICE, self).__init__()
        
        self.process_model_config(**graph_spice)
        # self.RETURNS.update(self.embedder.RETURNS)
        
        
    def process_model_config(self, 
                             skip_classes=[4], 
                             dimension=3, 
                             min_points=3, 
                             node_dim=22,
                             use_raw_features=False, 
                             constructor_cfg=None, 
                             embedder_cfg=None,
                             kernel_cfg=None,
                             invert=True,
                             use_true_labels=False,
                             make_fragments=False):
        
        if constructor_cfg is None:
            constructor_cfg = {}
        if embedder_cfg is None:
            embedder_cfg = {}
        if kernel_cfg is None:
            kernel_cfg = {}
        
        self.skip_classes     = skip_classes
        self.dimension        = dimension
        self.node_dim         = node_dim
        self.min_points       = min_points
        self.use_raw_features = use_raw_features
        self.use_true_labels  = use_true_labels
        self.make_fragments   = make_fragments
        self.invert           = invert
        
        self.embedder   = GraphSPICEEmbedder(embedder_cfg)
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)
        self.kernel_fn  = gs_kernel_factory(kernel_cfg)


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, 
                                         mode="fan_out", 
                                         nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def filter_class(self, input_tensors):
        '''
        Filter classes according to segmentation label.
        '''
        point_cloud, label = input_tensors
        mask = ~np.isin(label.tensor[:, -1].detach().cpu().numpy(), 
                        self.skip_classes)
        # valid_points, labels = point_cloud[mask], label[mask]
        valid_points = TensorBatch(point_cloud.tensor[mask], batch_size=input_tensors[0].batch_size, has_batch_col=True)
        labels = TensorBatch(label.tensor[mask], batch_size=input_tensors[1].batch_size, has_batch_col=True)
        return valid_points, labels
    

    def construct_fragments(self, input):
        
        raise NotImplementedError('Fragment construction not implemented.')
        
        frags = {}
        
        device = input[0].device
        semantic_labels = input[1][:, SHAPE_COL]
        filtered_semantic = ~(semantic_labels[..., None] == \
                                torch.tensor(self.skip_classes, device=device)).any(-1)
        graphs = self.gs_manager.fit_predict()
        perm = torch.argsort(graphs.voxel_id)
        cluster_predictions = graphs.node_pred[perm]
        filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                    semantic_labels[filtered_semantic].view(-1, 1),
                                    cluster_predictions.view(-1, 1)], dim=1)

        fragments = self._gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
        frags['filtered_input'] = [filtered_input]
        frags['fragment_batch_ids'] = [np.array(fragments[1])]
        frags['fragment_clusts'] = [np.array(fragments[0])]
        frags['fragment_seg'] = [np.array(fragments[2]).astype(int)]
        
        return frags


    def forward(self, input_data, cluster_label=None):
        '''Run a batch of data through the forward function.
        
        Parameters
        ----------
        input_data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
            
        cluster_label : TensorBatch, optional
            (N, 1 + D + 1 + N_labels) tensor of voxel/cluster label pairs. 

        '''
        
        # Pass input through the model
        
        input_tensors = [input_data, cluster_label]
        
        self.gs_manager.training = self.training
        
        valid_points, labels = self.filter_class(input_tensors)
        
        res = self.embedder(valid_points)

        res['coordinates'] = TensorBatch(valid_points.tensor[:, :COORD_COLS[-1]+1], 
                                         batch_size=input_data.batch_size, has_batch_col=True)
        
        if self.use_raw_features:
            res['hypergraph_features'] = res['features']

        # Build the graph
        # graph = self.gs_manager(res,
        #                         self.kernel_fn,
        #                         labels,
        #                         invert=self.invert)
        graph = self.gs_manager.initialize(res,
                                           labels,
                                           self.kernel_fn,
                                           invert=self.invert)
        
        # if self.make_fragments:
        #     frags = self.construct_fragments(valid_points)
        #     res.update(frags)
        
        graph_state = self.gs_manager.save_state()
        res.update(graph_state)

        return res


class GraphSPICELoss(nn.Module):
    """
    Loss function for GraphSpice.

    Configuration
    -------------
    name: str, default 'se_lovasz_inter'
        Loss function to use.
    invert: bool, default True
        You want to leave this to True for statistical weighting purpose.
    kernel_lossfn: str
    edge_loss_cfg: dict
        For example

        .. code-block:: yaml

          edge_loss_cfg:
            loss_type: 'LogDice'

    eval: bool, default False
        Whether we are in inference mode or not.

        .. warning::

            Currently you need to manually switch ``eval`` to ``True``
            when you want to run the inference, as there is no way (?)
            to know from within the loss function whether we are training
            or not.

    Output
    ------
    To be completed.

    See Also
    --------
    GraphSPICE
    """

    def __init__(self, graph_spice, graph_spice_loss=None):
        super(GraphSPICELoss, self).__init__()

        self.process_model_config(**graph_spice)
        self.process_loss_config(**graph_spice_loss)
        
        
    def process_model_config(self, skip_classes, invert=True, 
                             constructor_cfg=None, **kwargs):
        
        if constructor_cfg is None:
            constructor_cfg = {}
            
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)
        self.skip_classes = skip_classes
        self.invert = invert

        
    def process_loss_config(self, evaluate_true_accuracy=False,
                            name='se_lovasz_inter', **kwargs):

        self.evaluate_true_accuracy = evaluate_true_accuracy
        self.loss_fn = spice_loss_factory(name)(**kwargs)


    def filter_class(self, segment_label, cluster_label):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(segment_label[0][:, -1].cpu().numpy(), 
                        self.skip_classes)
        slabel = [segment_label[0][mask]]
        clabel = [cluster_label[0][mask]]
        return slabel, clabel


    def forward(self, segment_label, cluster_label, **result):
        '''

        '''
        self.gs_manager.load_state(result)
        
        slabel_tensor = [segment_label.tensor]
        clabel_tensor = [cluster_label.tensor]

        slabel, clabel = self.filter_class(slabel_tensor, clabel_tensor)
        
        res = self.loss_fn(result, slabel, clabel)
        
        if self.evaluate_true_accuracy:
            self.gs_manager.fit_predict()
            acc_out = self.gs_manager.evaluate()
            for key, val in acc_out.items():
                res[key] = val
                
        if 'ari' in res:
            res['accuracy'] = res['ari']
        return res
