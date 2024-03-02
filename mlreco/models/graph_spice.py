import torch
import numpy as np
import MinkowskiEngine as ME

from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.layers.cluster_cnn import gs_kernel_construct, spice_loss_construct

from mlreco.models.layers.cluster_cnn.graph_spice_embedder import GraphSPICEEmbedder

from pprint import pprint
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.unwrap import Unwrapper

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

    # RETURNS = {
    #     'image_id'     : ['tensor'],
    #     'coordinates'  : ['tensor'],
    #     'batch'        : ['tensor', 'image_id'],
    #     'x'            : ['tensor', 'image_id'],
    #     'pos'          : ['tensor', 'image_id'],
    #     'node_truth'   : ['tensor', 'image_id'],
    #     'voxel_id'     : ['tensor', 'image_id'],
    #     'graph_key'    : ['tensor'],
    #     'graph_id'     : ['tensor', 'graph_key'],
    #     'semantic_id'  : ['tensor', 'image_id'],
    #     'full_edge_index'   : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_index'   : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_batch'   : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_image_id': ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_label'   : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_attr'    : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_pred'    : ['edge_tensor', ['full_edge_index', 'image_id']],
    #     'edge_prob'    : ['edge_tensor', ['full_edge_index', 'image_id']]
    # }
    
    RETURNS = {
        'image_id': Unwrapper.Rule(method='tensor'),
        'coordinates': Unwrapper.Rule(method='tensor'),
        'batch': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        'x': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        'pos': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        'node_truth': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        'voxel_id': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        'graph_key': Unwrapper.Rule(method='tensor'),
        'graph_id': Unwrapper.Rule(method='tensor', ref_key='graph_key'),
        'semantic_id': Unwrapper.Rule(method='tensor', ref_key='image_id'),
        # TODO: Add other returns when unwrapper rules are implemented
    }

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
        self.RETURNS.update(self.embedder.RETURNS)
        
        
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
        self.kernel_fn  = gs_kernel_construct(kernel_cfg)


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, 
                                         mode="fan_out", 
                                         nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def filter_class(self, input):
        '''
        Filter classes according to segmentation label.
        '''
        point_cloud, label = input
        mask = ~np.isin(label[:, -1].detach().cpu().numpy(), 
                        self.skip_classes)
        x = [point_cloud[mask], label[mask]]
        return x


    def forward(self, input_data, cluster_label=None):
        '''

        '''
        
        print(input_data)
        print(cluster_label)
        assert False
        
        # Pass input through the model
        self.gs_manager.training = self.training
        point_cloud, labels = self.filter_class(input)
        res = self.embedder([point_cloud])

        res['coordinates'] = [point_cloud[:, :4]]
        if self.use_raw_features:
            res['hypergraph_features'] = res['features']

        # Build the graph
        graph = self.gs_manager(res,
                                self.kernel_fn,
                                labels,
                                invert=self.invert)
        
        graph_state = self.gs_manager.save_state(unwrapped=False)
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

    RETURNS = {
        'loss': Unwrapper.Rule(method='scalar'),
        'accuracy': Unwrapper.Rule(method='scalar'),
    }

    def __init__(self, graph_spice, graph_spice_loss=None):
        super(GraphSPICELoss, self).__init__()

        self.process_model_config(**graph_spice)
        self.process_loss_config(**graph_spice_loss)

        self.RETURNS.update(self.loss_fn.RETURNS)
        
        
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
        self.loss_fn = spice_loss_construct(name)(**kwargs)


    def filter_class(self, segment_label, cluster_label):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(segment_label[0][:, -1].cpu().numpy(), 
                        self.skip_classes)
        slabel = [segment_label[0][mask]]
        clabel = [cluster_label[0][mask]]
        return slabel, clabel


    def forward(self, result, segment_label, cluster_label):
        '''

        '''
        self.gs_manager.load_state(result, unwrapped=False)

        # if self.invert:
        #     pred_labels = result['edge_score'][0] < 0.0
        # else:
        #     pred_labels = result['edge_score'][0] >= 0.0
        # edge_diff = pred_labels != (result['edge_truth'][0] > 0.5)

        slabel, clabel = self.filter_class(segment_label, cluster_label)
        res = self.loss_fn(result, slabel, clabel)
        
        if self.evaluate_true_accuracy:
            self.gs_manager.fit_predict()
            acc_out = self.gs_manager.evaluate()
            for key, val in acc_out.items():
                res[key] = val
        return res
