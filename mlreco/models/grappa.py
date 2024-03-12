import random
import torch
import numpy as np

from .layers.common.dbscan import DBSCANFragmenter
from .layers.common.momentum import (
        DeepVertexNet, EvidentialMomentumNet, MomentumNet, VertexNet)
from .layers.gnn import (
        graph_construct, gnn_model_construct, node_encoder_construct,
        edge_encoder_construct, node_loss_construct, edge_loss_construct)

from .experimental.transformers.transformer import TransformerEncoderLayer

from mlreco.utils.globals import (
        BATCH_COL, COORD_COLS, CLUST_COL, GROUP_COL, SHAPE_COL, LOWES_SHP)
from mlreco.utils.gnn.data import merge_batch, split_clusts, split_edge_index
from mlreco.utils.gnn.cluster import (
        form_clusters, get_cluster_batch, get_cluster_label, 
        get_cluster_primary_label)

__all__ = ['GrapPA', 'GrapPALoss']


class GrapPA(torch.nn.Module):
    """Graph Particle Aggregator (GrapPA) model.

    This class mostly acts as a wrapper that will hand the graph data
    to the underlying graph neural network (GNN).

    When trained standalone, this model must be provided with a cluster
    label tensor, allowing it to build a set of intput clusters based on the
    label boundaries of the clusters and their semantic types.

    Typical configuration can look like this:

    .. code-block:: yaml

        model:
          name: grappa
          modules:
            grappa:
              # Your config goes here

    Configuration
    -------------
    base: dict
        Configuration of base Grappa :

        .. code-block:: yaml

          base:
            source      : <column in the input data that specifies the source node ids of each voxel (default 5)>
            target      : <column in the input data that specifies the target instance ids of each voxel (default 6)>
            node_type       : <semantic class to aggregate (all classes if -1, default -1)>
            node_min_size   : <minimum number of voxels inside a cluster to be included in the aggregation (default -1, i.e. no threshold)>
            add_points      : <add label point(s) to the node features: False (none) or True (both) (default False)>
            add_local_dirs  : <add reconstructed local direction(s) to the node features: False (none), True (both) or 'start' (default False)>
            dir_max_dist    : <maximium distance between start point and cluster voxels to be used to estimate direction: support value or 'optimize' (default 5 voxels)>
            add_local_dedxs : <add reconstructed local dedx(s) to the node features: False (none), True (both) or 'start' (default False)>
            dedx_max_dist   : <maximium distance between start point and cluster voxels to be used to estimate dedx (default 5 voxels)>
            network         : <type of network: 'complete', 'delaunay', 'mst', 'knn' or 'bipartite' (default 'complete')>
            edge_max_dist   : <maximal edge Euclidean length (default -1)>
            edge_dist_method: <edge length evaluation method: 'centroid' or 'voxel' (default 'voxel')>
            merge_batch     : <flag for whether to merge batches (default False)>
            merge_batch_mode: <mode of batch merging, 'const' or 'fluc'; 'const' use a fixed size of batch for merging, 'fluc' takes the input size a mean and sample based on it (default 'const')>
            merge_batch_size: <size of batch merging (default 2)>

    dbscan: dict

        dictionary of dbscan parameters

    node_encoder: dict

        .. code-block:: yaml

          node_encoder:
            name: <name of the node encoder>
            <dictionary of arguments to pass to the encoder>
            model_path      : <path to the encoder weights>

    edge_encoder: dict

        .. code-block:: yaml

          edge_encoder:
            name: <name of the edge encoder>
            <dictionary of arguments to pass to the encoder>
            model_path      : <path to the encoder weights>

    gnn_model: dict
        .. code-block:: yaml

          gnn_model:
            name: <name of the node model>
            <dictionary of arguments to pass to the model>
            model_path      : <path to the model weights>

    kinematics_mlp: bool, default False
        Whether to enable MLP-like layers after the GrapPA to predict
        momentum, particle type, etc.
    kinematics_type: bool
        Whether to add PID MLP to each node.
    kinematics_momentum: bool
        Whether to add momentum MLP to each node.
    type_net: dict
        Configuration for the PID MLP (if enabled).
        Can partial load weights here too.
    momentum_net: dict
        Configuration for the Momentum MLP (if enabled).
        Can partial load weights here too.
    vertex_mlp: bool, default False
        Whether to add vertex prediction MLP to each node.
        Includes primary particle + vertex coordinates predictions.
    vertex_net: dict
        Configuration for the Vertex MLP (if enabled).
        Can partial load weights here too.

    Outputs
    -------
    node_features:
    edge_features:
    clusts:
    edge_index:
    node_pred:
    edge_pred:
    node_pred_p:
    node_pred_type:
    node_pred_vtx:

    See Also
    --------
    GrapPALoss
    """

    MODULES = [('grappa', ['base', 'dbscan', 'node_encoder', 'edge_encoder', 'gnn_model']), 'grappa_loss']

    RETURNS = {
        'clusts' : ['index_list', ['input_data', 'batch_ids'], True], # TODO
        'node_features': ['tensor', 'batch_ids', True],
        'node_pred': ['tensor', 'batch_ids', True],
        'node_pred_type': ['tensor', 'batch_ids', True],
        'node_pred_vtx': ['tensor', 'batch_ids', True],
        'node_pred_p': ['tensor', 'batch_ids', True],
        'start_points': ['tensor', 'batch_ids', False, True],
        'end_points': ['tensor', 'batch_ids', False, True],
        'group_pred': ['index_tensor', 'batch_ids', True],
        'edge_features': ['edge_tensor', ['edge_index', 'batch_ids'], True],
        'edge_index': ['edge_tensor', ['edge_index', 'batch_ids'], True],
        'edge_pred': ['edge_tensor', ['edge_index', 'batch_ids'], True]
    }

    def __init__(self, grappa, grappa_loss=None):
        """Initialize the GrapPA model.

        Parameters
        ----------
        grappa : dict
            Model configuration
        grappa_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the model configuration
        self.process_model_config(**grappa)

    def process_model_config(self, nodes, graph, node_encoder,
                             edge_encoder, gnn_model, dbscan=None):
        """Process the top-level configuration block.

        This dispatches each block to its own configuration processor.

        Parameters
        ----------
        nodes : dict
            Input node configuration
        graph : dict
            Graph configuration
        node_encoder : dict
            Node encoder configuration
        edge_encoder : dict
            Edge encoder configuration
        gnn_model : dict
            Underlying graph neural network configuration
        dbscan : dict
            DBSCAN fragmentation configuration
        """
        # Process the node configuration
        self.process_node_config(**nodes)

        # Process the graph configuration
        graph['classes'] = self.node_type
        self.graph_constructor = graph_construct(graph)

        # Process the node encoder configuration
        self.node_encoder = node_encoder_construct(node_encoder)

        # Initialize edge encoder
        self.edge_encoder = edge_encoder_construct(edge_encoder)

        # Construct the underlying graph neural network
        self.gnn_model = gnn_model_construct(gnn_model)

        # Process the dbscan fragmenter configuration, if provided
        if dbscan is not None:
            self.process_dbscan_config(dbscan)

        # Construct the node prediction layer, if specified
        #TODO

    def process_node_config(self, source=CLUST_COL, 
                            semantic_class=-1, min_size=-1):
        """Process the node parameters of the model.

        Parameters
        ----------
        source : int, default CLUST_COL
            Column in the label tensor which contains the input cluster IDs
        class : int, default -1
            Type of nodes to include in the input. If -1, include all types
        min_size : int, default -1
            Minimum number of voxels in a cluster to be included in the input
        """
        # Store the node parameters
        self.node_source   = source
        self.node_type     = semantic_class
        self.node_min_size = min_size

        # Interpret node type as list of classes to cluster
        if isinstance(semantic_class, int):
            if semantic_class == -1:
                self.node_type = list(np.arange(LOWES_SHP))
            else:
                self.node_type = [semantic_class]

    def process_dbscan_config(cluster_classes=None, min_size=None, **kwargs):
        """Process the DBSCAN fragmenter configuration.

        Parameters
        ----------
        cluster_classes : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        min_size : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        **kwargs : dict, optional
            Rest of the DBSCAN configuration
        """
        # Make sure the basic parameters are not specified twice
        assert cluster_classes is not None and min_size is not None, (
                "Do not specify 'cluster_classes' or 'min_size' in the "
                "`dbscan` block, it is shared with the `node` block")

        # Initialize DBSCAN fragmenter
        self.dbscan = DBSCANFragmenter(
                cluster_classes=self.node_type,
                min_size=self.min_size, **kwargs)

    def process_(cluster_classes=None, min_size=None, **kwargs):
        """Process the DBSCAN fragmenter configuration.

        Parameters
        ----------
        cluster_classes : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        min_size : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        **kwargs : dict, optional
            Rest of the DBSCAN configuration
        """
        # TODO: Make a final layer factory elsewhere to call upon here

        # If requested, initialize two MLPs for kinematics predictions
        self.kinematics_mlp = base_config.get('kinematics_mlp', False)
        self.kinematics_type = base_config.get('kinematics_type', False)
        self.kinematics_momentum = base_config.get('kinematics_momentum', False)
        if self.kinematics_mlp:
            node_output_feats = cfg[name]['gnn_model'].get('node_output_feats', 64)
            self.kinematics_type = base_config.get('kinematics_type', False)
            self.kinematics_momentum = base_config.get('kinematics_momentum', False)
            if self.kinematics_type:
                type_config = cfg[name].get('type_net', {})
                type_net_mode = type_config.get('mode', 'standard')
                type_net_num_classes = type_config.get('num_classes', 5)
                if type_net_mode == 'linear':
                    self.type_net = torch.nn.Linear(node_output_feats, type_net_num_classes)
                elif type_net_mode == 'standard':
                    self.type_net = MomentumNet(node_output_feats,
                                                num_output=type_net_num_classes,
                                                num_hidden=type_config.get('num_hidden', 128),
                                                positive_outputs=type_config.get('positive_outputs', False))
                elif type_net_mode == 'edl':
                    self.type_net = MomentumNet(node_output_feats,
                                                num_output=type_net_num_classes,
                                                num_hidden=type_config.get('num_hidden', 128),
                                                positive_outputs=type_config.get('positive_outputs', True))
                else:
                    raise ValueError('Unrecognized Particle ID Type Net Mode: ', type_net_mode)
            if self.kinematics_momentum:
                momentum_config = cfg[name].get('momentum_net', {})
                softplus_and_shift = momentum_config.get('eps', 0.0)
                logspace = momentum_config.get('logspace', False)
                if momentum_config.get('mode', 'standard') == 'edl':
                    self.momentum_net = EvidentialMomentumNet(node_output_feats,
                                                              num_output=4,
                                                              num_hidden=momentum_config.get('num_hidden', 128),
                                                              eps=softplus_and_shift,
                                                              logspace=logspace)
                else:
                    self.momentum_net = MomentumNet(node_output_feats,
                                                    num_output=1,
                                                    num_hidden=momentum_config.get('num_hidden', 128))

        self.vertex_mlp = base_config.get('vertex_mlp', False)
        if self.vertex_mlp:
            node_feats = cfg[name]['gnn_model'].get('node_feats')
            node_output_feats = cfg[name]['gnn_model'].get('node_output_feats')
            vertex_config = cfg[name].get('vertex_net', {'name': 'momentum_net'})
            self.pred_vtx_positions = vertex_config.get('pred_vtx_positions', True)
            self.use_vtx_input_features = vertex_config.get('use_vtx_input_features', False)
            self.add_vtx_input_features = vertex_config.get('add_vtx_input_features', False)
            num_input  = node_output_feats + node_feats * self.add_vtx_input_features 
            num_output = 2 + 3 * self.pred_vtx_positions
            vertex_net_name = vertex_config.get('name', 'momentum_net')
            if vertex_net_name == 'linear':
                self.vertex_net = torch.nn.Linear(num_input, num_output)
            elif vertex_net_name == 'momentum_net':
                self.vertex_net = VertexNet(num_input, num_output,
                                            num_hidden=vertex_config.get('num_hidden', 64),
                                            positive_outputs=vertex_config.get('positive_outputs',False))
            elif vertex_net_name == 'attention_net':
                self.vertex_net = TransformerEncoderLayer(num_input, num_output, **vertex_config)
            elif vertex_net_name == 'deep_vertex_net':
                self.vertex_net = DeepVertexNet(num_input, num_output,
                                                num_hidden=vertex_config.get('num_hidden', 64),
                                                num_layers=vertex_config.get('num_layers', 5),
                                                positive_outputs=vertex_config.get('positive_outputs',False))
            else:
                raise ValueError('Vertex MLP {} not recognized!'.format(vertex_config['name']))


    def forward(self, data, part_coords=None, clusts=None, classes=None,
                groups=None, points=None, extra_feats=None):
        """Prepares particle clusters and feed them to the GNN model.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is 1 (charge/energy) if the clusters (`clusts`) are provided,
              or it needs to contain cluster labels to build them on the fly
        part_coords : TensorBatch, optional
            (P, 1 + 2*D + 2) Tensor of label points (start/end/time/shape)
        clusts : IndexListBatch, optional
            (C) List of indexes corresponding to each cluster
        classes : np.ndarray, optional
            (C) List of cluster semantic class used to define the max length
        groups : TensorBatch, optional
            (C) List of node groups, one per cluster. If specified, will
                remove connections between nodes of a separate group.
        points : TensorBatch, optional
            (C, 3/6) Tensor of start (and end) points (TODO: merge with part_coords?)
        extra_feats : TensorBatch, optional
            (C, N_f) Batch of features to append to the existing node features
            extra_feats: (N,F) tensor of features to add to the encoded features

        Returns
        -------
        clusts : IndexBatch
            (C, N_c, N_{c,i}) Cluster indexes
        edge_index : TensorBatch
            (E, 2) Incidence matrix
        node_pred : TensorBatch
            (C, N_n) Node predictions (logits)
        edge_pred : TensorBatch
            (C, N_e) Edge predictions (logits)

        TODO: CONTINUE
        """
        # Cast the labels to numpy for the functions run on CPU
        data_np = data.to_numpy()
        part_coords_np = None
        if part_coords is not None:
            part_coords_np = part_coords.to_numpy()

        # If not provided, form the clusters: a list of list of voxel indices,
        # one list per cluster matching the list of requested class
        if clusts is None:
            if hasattr(self, 'dbscan'):
                # Use the DBSCAN fragmenter to build the clusters
                clusts = self.dbscan(data_np, part_coords_np)
            else:
                # Use the label tensor to build the clusters
                clusts = form_clusters(
                        data_np, self.node_min_size,
                        self.node_source, cluster_classes=self.node_type)

        # If the graph edge length cut is class-specific, get the class labels
        if (classes is None and 
            hasattr(self.graph_constructor.max_length, '__len__')):
            if self.node_source == GROUP_COL:
                # For groups, use primary shape to handle Michel/Delta properly
                classes = get_cluster_primary_label(
                        data_np.tensor, clusts.index_list, SHAPE_COL)
            else:
                # Just use the shape of the cluster itself otherwise
                classes = get_cluster_label(
                        data_np.tensor, clusts.index_list, SHAPE_COL)

            classes = classes.astype(np.int64)

        # Initialize the input graph
        edge_index, dist_mat, closest_index = self.graph_constructor(
                data_np, clusts, classes, groups)

        # Obtain node and edge features
        # TODO: Make kwargs out of non-None variables
        x = self.node_encoder(cluster_data, clusts)
        e = self.edge_encoder(
                cluster_data, clusts, edge_index, closest_index=closest_index)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=data.tensor.device, dtype=torch.long)
        xbatch = torch.tensor(clusts.batch_ids, device=data.tensor.device)

        # Pass through the model, update results
        out = self.gnn_model(x, index, e, xbatch)

        # Build the result dictionary
        result = {
            "clusts": clusts,
            "edge_index": edge_index,
            "node_features": node_features,
            "edge_features": edge_features,
            "node_pred": out['node_pred'],
            "edge_pred": out['edge_pred']
        }

        # If requested, pass the node features through additional MLPs
        if self.kinematics_mlp:
            if self.kinematics_type:
                node_pred_type = self.type_net(out['node_features'][0])
                result['node_pred_type'] = [[node_pred_type[b] for b in cbids]]
            if self.kinematics_momentum:
                node_pred_p = self.momentum_net(out['node_features'][0])
                if isinstance(self.momentum_net, EvidentialMomentumNet):
                    result['node_pred_p'] = [[node_pred_p[b] for b in cbids]]
                    aleatoric = node_pred_p[:, 3] / (node_pred_p[:, 2] - 1.0 + 0.001)
                    epistemic = node_pred_p[:, 3] / (node_pred_p[:, 1] * (node_pred_p[:, 2] - 1.0 + 0.001))
                    result['node_pred_p_aleatoric'] = [[aleatoric[b] for b in cbids]]
                    result['node_pred_p_epistemic'] = [[epistemic[b] for b in cbids]]
                else:
                    result['node_pred_p'] = [[node_pred_p[b] for b in cbids]]
        else:
            # If final post-gnn MLP is not given, set type features to node_pred.
            result['node_pred_type'] = result['node_pred']

        if self.vertex_mlp:
            if self.use_vtx_input_features:
                node_pred_vtx = self.vertex_net(x)
            elif self.add_vtx_input_features:
                node_pred_vtx = self.vertex_net(torch.cat([x, out['node_features'][0]], dim=1))
            else:
                node_pred_vtx = self.vertex_net(out['node_features'][0])
            result['node_pred_vtx'] = [[node_pred_vtx[b] for b in cbids]]


        return result


class GrapPALoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of the GrapPA and computes the total loss.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: <name of the node loss>
                <dictionary of arguments to pass to the loss>
              edge_loss:
                name: <name of the edge loss>
                <dictionary of arguments to pass to the loss>
    """

    RETURNS = {
        'loss': ['scalar'],
        'node_loss': ['scalar'],
        'edge_loss': ['scalar'],
        'accuracy': ['scalar'],
        'node_accuracy': ['scalar'],
        'edge_accuracy': ['scalar']
    }

    def __init__(self, grappa, grappa_loss):
        """Initialize the GrapPA loss function.

        Parameters
        ----------
        grappa : dict
            Model configuration
        grappa_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the model configuration
        #self.process_model_config(**grappa)

        # Process the loss configuration
        self.process_loss_config(**grappa_loss)

    def process_loss_config(self, node_loss=None, edge_loss=None):
        """Process the loss configuration.

        Parameters
        ----------
        node_loss : Union[dict, Dict[dict]], optional
            Node loss configuration
        edge_loss : Union[dict, Dict[dict]], optional
            Edge loss configuration
        """
        # Check that there is at least one loss to apply
        assert node_loss is not None or edge_loss is not None, (
                "Must provide either a `node_loss` or `edge_loss` to the "
                "GrapPA loss function.")

        # Initialize the loss components
        # TODO: add support for a dictionary of losses!
        self.apply_node_loss, self.apply_edge_loss = False, False
        if node_loss is not None:
            self.apply_node_loss = True
            self.node_loss = node_loss_construct(node_loss)
            self.RETURNS.update(self.node_loss.RETURNS) # TODO remove

        if edge_loss is not None:
            self.apply_edge_loss = True
            self.edge_loss = edge_loss_construct(edge_loss)
            self.RETURNS.update(self.edge_loss.RETURNS) # TODO remove


    def forward(self, result, clust_label, graph=None, node_label=None, iteration=None):

        # Apply edge and node losses, if instantiated
        loss = {}
        if self.apply_node_loss:
            if node_label is None:
                node_label = clust_label
            if iteration is not None:
                node_loss = self.node_loss(result, node_label, iteration=iteration)
            else:
                node_loss = self.node_loss(result, node_label)
            loss.update(node_loss)
            loss['node_loss'] = node_loss['loss']
            loss['node_accuracy'] = node_loss['accuracy']
        if self.apply_edge_loss:
            edge_loss = self.edge_loss(result, clust_label, graph)
            loss.update(edge_loss)
            loss['edge_loss'] = edge_loss['loss']
            loss['edge_accuracy'] = edge_loss['accuracy']
        if self.apply_node_loss and self.apply_edge_loss:
            loss['loss'] = loss['node_loss'] + loss['edge_loss']
            loss['accuracy'] = (loss['node_accuracy'] + loss['edge_accuracy'])/2

        return loss
