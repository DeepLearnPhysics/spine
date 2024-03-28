"""Module with the core full reconstruction chain."""

import yaml
import numpy as np
import torch
from typing import Union, List

from .uresnet import UResNetSegmentation, SegmentationLoss
from .uresnet_ppn import UResNetPPN, UResNetPPNLoss
from .graph_spice import GraphSPICE, GraphSPICELoss
from .grappa import GrapPA, GrapPALoss
from .layers.common.dbscan import DBSCAN

#from mlreco.models.layers.cnn.cnn_encoder import SparseResidualEncoder ??
# TODO: replace this with MultiParticleImageClassifier
# TODO: raname it something more generic like ParticleClusterImageClassifier?

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.utils.ppn import get_particle_points
from mlreco.utils.ghost import compute_rescaled_charge, adapt_labels
from mlreco.utils.cluster.fragmenter import (
        GraphSPICEFragmentManager, format_fragments)
from mlreco.utils.gnn.cluster import get_cluster_features_extended
from mlreco.utils.logger import logger


class FullChain(torch.nn.Module):
    """Full reconstruction in all its glory.

    Modular, end-to-end particle imaging detector reconstruction chain:
    - Deghosting for 3D tomographic reconstruction artifiact removal
    - Voxel-wise semantic segmentation
    - Point proposal
    - Particle clustering
    - Shower primary identification
    - Interaction clustering
    - Particle type classification
    - Primary identification
    - Track orientation

    Typical configuration can look like this:

    .. code-block:: yaml

        model:
          name: grappa
          modules:
            chain:
               <dictionary of arguments to specify chain-wide configuration>
            uresnet_deghost:
               name: <name of the model used to deghost
               <dictionary of arguments to specify the deghosting module>
            uresnet_segmentation:
               name: <name of the model used to do segmentation>
               <dictionary of arguments to specify the segmentation module>
            dbscan:
               TODO
            graph_spice:
               name: <name of the model used to do segmentation>
               <dictionary of arguments to specify the segmentation module>
            grappa_shower:
               name: <name of the model used to do segmentation>
               <dictionary of arguments to specify the segmentation module>
            grappa_track:
               name: <name of the model used to do segmentation>
               <dictionary of arguments to specify the segmentation module>
            grappa_inter:
              TODO
            calibration:
              TODO

    See configuration file(s) prefixed with `full_chain_` under the `config`
    directory for detailed examples of working configurations.

    The ``chain`` section enables or disables specific stages of the full
    chain. When a module is disabled through this section, it will not be
    constructed. The configuration blocks for each enabled module should
    also live under the `modules` section of the configuration.
    """
    # TODO: update
    MODULES = ['grappa_shower', 'grappa_track', 'grappa_inter',
               'grappa_shower_loss', 'grappa_track_loss', 'grappa_inter_loss',
               'full_chain_loss', 'graph_spice', 'graph_spice_loss',
               'fragment_clustering',  'chain', 'dbscan_frag',
               ('uresnet_ppn', ['uresnet', 'ppn'])]

    # Store the valid chain modes
    modes = {
            'deghosting': ['uresnet'],
            'charge_rescaling': ['uresnet'],
            'segmentation': ['uresnet'],
            'point_proposal': ['ppn'],
            'dense_clustering': ['dbscan', 'graph_spice', 'dbscan_graph_spice'],
            'shower_aggregation': ['grappa'],
            'shower_primary': ['grappa'],
            'track_aggregation': ['grappa'],
            'particle_aggregation': ['grappa'],
            'inter_aggregation': ['grappa'],
            'particle_identification': ['grappa', 'image'],
            'primary_identification': ['grappa'],
            'orientation_identification': ['grappa'],
            'calibration': ['default']
    }

    def __init__(self, chain, uresnet_deghost=None, uresnet=None,
                 uresnet_ppn=None, graph_spice=None, dbscan=None,
                 grappa_shower=None, grappa_track=None, grappa_particle=None,
                 grappa_inter=None, calibration=None, uresnet_deghost_loss=None,
                 uresnet_loss=None, uresnet_ppn_loss=None,
                 graph_spice_loss=None, grappa_shower_loss=None,
                 grappa_track_loss=None, grappa_particle_loss=None,
                 grappa_inter_loss=None):
        """Initialize the full chain model.

        Parameters
        ----------
        chain : dict
            Dictionary of parameters used to configure the chain
        uresnet_deghost : dict, optional
            Deghosting model configuration
        uresnet_ppn : dict, optional
            Segmentation and point proposal model configuration
        dbscan : dict, optional
            Connected component clustering configuration
        graph_spice : dict, optional
            Supervised connected component clustering model configuration
        grappa_shower : dict, optional
            Shower aggregation model configuration
        grappa_track : dict, optional
            Track aggregation model configuration
        grappa_particle : dict, optional
            Global particle aggregation configuration
        grappa_inter : dict, optional
            Interaction aggregation model configuration
        calibration : dict, optional
            Calibration configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the main chain configureation
        process_chain_config(self, **chain, dump_config=True)

        # Initialize the deghosting model
        if self.deghosting is not None and self.deghosting == 'uresnet':
            assert uresnet_deghost is not None, (
                    "If the deghosting is using UResNet, must provide the "
                    "`uresnet_deghost` configuration block.")
            self.uresnet_deghost = UResNetSegmentation(uresnet_deghost)
            self.deghost_num_input = self.uresnet_deghost.backbone.num_input

        # Initialize the semantic segmentation model (+ point proposal)
        if self.segmentation is not None and self.segmentation == 'uresnet':
            assert (uresnet is not None) ^ (uresnet_ppn is not None), (
                    "If the segmentation is using UResNet, must provide the "
                    "`uresnet` or `uresnet_ppn` configuration block.")
            if uresnet is not None:
                self.uresnet = UResNetSegmentation(uresnet)
                self.seg_num_input = self.uresnet.backbone.num_input
            else:
                self.uresnet_ppn = UResNetPPN(**uresnet_ppn)
                self.seg_num_input = self.uresnet_ppn.uresnet.backbone.num_input

        # Initialize the dense clustering model
        self.fragment_classes = []
        if self.dense_clustering is not None:
            if 'dbscan' in self.dense_clustering:
                assert dbscan is not None, (
                        "If the fragmentation is done using DBSCAN, must "
                        "provide the `dbscan` configuration block.")
                self.dbscan = DBSCAN(**dbscan)
                self.fragment_classes.extend(self.dbscan.classes)
            if 'graph_spice' in self.dense_clustering:
                assert graph_spice is not None, (
                        "If the fragmentation is done using Graph-SPICE, must "
                        "provide the `graph_spice` configuration block.")
                self.graph_spice = GraphSPICE(graph_spice)
                self.fragment_classes.extend(self.graph_spice.classes)

            # Check that there is no redundancy between the fragmenters
            uniques, counts = np.unique(
                    self.fragment_classes, return_counts=True)
            assert np.all(uniques == np.arange(4)), (
                    "All four expected semantic classes should be fragmented "
                    "by either DBSCAN or Graph-SPICE.")
            assert np.all(counts) == 1, (
                    "Some of the classes appear in both the DBSCAN and the "
                    "Graph-SPICE based fragmentation. Ambiguous.")

        # Initialize the graph-based aggregator modules
        grappa_configs = {'shower': grappa_shower, 'track': grappa_track,
                          'particle': grappa_particle, 'inter': grappa_inter}
        for stage, config in grappa_configs.items():
            if getattr(self, f'{stage}_aggregation') == 'grappa':
                name = f'grappa_{stage}'
                assert config is not None, (
                        f"If the {stage} aggregation is done using GrapPA, "
                        f"must provide the {name} configuration block.")
                setattr(self, name, GrapPA(config))

        # Initialize the interaction-classification module
        # TODO (could be done by either CNN or graph-level GNN)

    def forward(self, data, segment_label=None, clust_label=None):
        """Run a batch of data through the full chain.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        segment_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch
        clust_label : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is is the number of cluster labels

        Returns
        -------
        TODO
        """
        # Initialize the full chain output dictionary
        result = {}

        # Run the deghosting
        deghost_result = self.run_deghosting(data, segment_label)
        result.update(self.run_deghosting())

    def run_deghosting(self, data, segment_label=None):
        """Run the deghosting algorithm.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        segment_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch

        Returns
        -------
        ghost : TensorBatch
        """
        if self.deghosting == 'uresnet':
            # Restrict the input to the required number of features
            pass
        elif self.deghosting == 'label':
            # Use ghost labels to remove ghost voxels from the input
            assert segment_label is not None, (
                    "Must provide `segment_label` to deghost using the it.")
        else:
            # If there is no deghosting to do, return an empty dictionary
            return {}


        deghost = None
        if self.enable_charge_rescaling:
            # Pass through the deghosting
            assert self.enable_ghost
            last_index = 4 + self.deghost_input_features
            result.update(self.uresnet_deghost([input[0][:,:last_index]]))
            result['ghost'] = result['segmentation']
            deghost = result['ghost'][0][:, 0] > result['ghost'][0][:,1]
            del result['segmentation']

            # Rescale the charge column, store it
            charges = compute_rescaled_charge(input[0], deghost, last_index=last_index, collection_only=self.collection_charge_only)
            input[0][deghost, VALUE_COL] = charges

            input_rescaled = input[0][deghost,:5].clone()
            input_rescaled[:, VALUE_COL] = charges

            result.update({'input_rescaled':[input_rescaled]})
            if input[0].shape[1] == (last_index + 6 + 2):
                result.update({'input_rescaled_source':[input[0][deghost,-2:]]})

        if self.enable_uresnet:
            if not self.enable_charge_rescaling:
                result.update(self.uresnet_lonely([input[0][:, :4+self.input_features]]))
            else:
                if torch.sum(deghost):
                    result.update(self.uresnet_lonely([input[0][deghost, :4+self.input_features]]))
                else:
                    # TODO: move empty case handling elsewhere
                    seg = torch.zeros((input[0][deghost,:5].shape[0], 5), device=input[0].device, dtype=input[0].dtype) # DUMB
                    result['segmentation'] = [seg]
                    return result, input

        if self.enable_ppn:
            ppn_input = {}
            ppn_input.update(result)
            if 'ghost' in ppn_input and not self.enable_charge_rescaling:
                ppn_input['ghost'] = ppn_input['ghost'][0]
                ppn_output = self.ppn(ppn_input['finalTensor'][0],
                                      ppn_input['decoderTensors'][0],
                                      ppn_input['ghost_tensor'][0])
            else:
                ppn_output = self.ppn(ppn_input['finalTensor'][0],
                                      ppn_input['decoderTensors'][0])
            result.update(ppn_output)

        # The rest of the chain only needs 1 input feature
        if self.input_features > 1:
            input[0] = input[0][:, :-self.input_features+1]

        cnn_result = {}

        if label_seg is not None and label_clustering is not None:
            label_clustering = [adapt_labels(label_clustering[0],
                                             label_seg[0],
                                             result['segmentation'][0],
                                             deghost)]

        if self.enable_ghost:
            # Update input based on deghosting results
            # if self.cheat_ghost:
            #     assert label_seg is not None
            #     deghost = label_seg[0][:, self.uresnet_lonely.ghost_label] == \
            #               self.uresnet_lonely.num_classes
            #     print(deghost, deghost.shape)
            # else:
            deghost = result['ghost'][0][:,0] > result['ghost'][0][:,1]

            input = [input[0][deghost]]

            deghost_result = {}
            deghost_result.update(result)
            deghost_result.pop('ghost')
            if self.enable_ppn and not self.enable_charge_rescaling:
                deghost_result['ppn_points'] = [result['ppn_points'][0][deghost]]
                deghost_result['ppn_masks'][0][-1]  = result['ppn_masks'][0][-1][deghost]
                deghost_result['ppn_coords'][0][-1] = result['ppn_coords'][0][-1][deghost]
                deghost_result['ppn_layers'][0][-1] = result['ppn_layers'][0][-1][deghost]
                if 'ppn_classify_endpoints' in deghost_result:
                    deghost_result['ppn_classify_endpoints'] = [result['ppn_classify_endpoints'][0][deghost]]
            cnn_result.update(deghost_result)
            cnn_result['ghost'] = result['ghost']

        else:
            cnn_result.update(result)


        # ---
        # 1. Clustering w/ CNN or DBSCAN will produce
        # - fragments (list of list of integer indexing the input data)
        # - fragments_batch_ids (list of batch ids for each fragment)
        # - fragments_seg (list of integers, semantic label for each fragment)
        # ---

        cluster_result = {
            'fragment_clusts': [],
            'fragment_batch_ids': [],
            'fragment_seg': []
        }
        if self._gspice_use_true_labels:
            semantic_labels = label_seg[0][:, -1]
        else:
            semantic_labels = torch.argmax(cnn_result['segmentation'][0], dim=1).flatten()
            if not self.enable_charge_rescaling and 'ghost' in cnn_result:
                deghost = result['ghost'][0].argmax(dim=1) == 0
                semantic_labels = semantic_labels[deghost]

        if self.enable_cnn_clust:
            if label_clustering is None and self.training:
                raise Exception("Cluster labels from parse_cluster3d_clean_full are needed at this time for training.")

            filtered_semantic = ~(semantic_labels[..., None] == \
                                    torch.tensor(self._gspice_skip_classes, device=device)).any(-1)

            # If there are voxels to process in the given semantic classes
            if torch.count_nonzero(filtered_semantic) > 0:
                if label_clustering is not None:
                    # If labels are present, compute loss and accuracy
                    graph_spice_label = torch.cat((label_clustering[0][:, :-1],
                                                    semantic_labels.reshape(-1,1)), dim=1)
                else:
                #     # Otherwise run in data inference mode (will not compute loss and accuracy)
                    graph_spice_label = torch.cat((input[0][:, :4],
                                                    semantic_labels.reshape(-1, 1)), dim=1)
                cnn_result['graph_spice_label'] = [graph_spice_label]
                spatial_embeddings_output = self.graph_spice((input[0][:,:5],
                                                              graph_spice_label))
                cnn_result.update({f'graph_spice_{k}':v for k, v in spatial_embeddings_output.items()})

                if self.process_fragments:

                    self.gs_manager.load_state(spatial_embeddings_output)   
                    
                    graphs = self.gs_manager.fit_predict(min_points=self._gspice_min_points)
                    
                    perm = torch.argsort(graphs.voxel_id)
                    cluster_predictions = graphs.node_pred[perm]

                    filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                                semantic_labels[filtered_semantic].view(-1, 1),
                                                cluster_predictions.view(-1, 1)], dim=1)
                    
                    # For the record - (self.gs_manager._node_pred.pos == input[0][filtered_semantic][:, 1:4]).all()
                    # ie ordering of voxels is NOT the same in node predictions and (filtered) input data
                    # It is likely that input data is lexsorted while node predictions 
                    # (and anything that are concatenated through Batch.from_data_list) are not. 

                    fragment_data = self._gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
                    cluster_result['fragment_clusts'].extend(fragment_data[0])
                    cluster_result['fragment_batch_ids'].extend(fragment_data[1])
                    cluster_result['fragment_seg'].extend(fragment_data[2])

        if self.enable_dbscan and self.process_fragments:
            # Get the fragment predictions from the DBSCAN fragmenter
            fragment_data = self.dbscan_fragment_manager(input[0], cnn_result)
            cluster_result['fragment_clusts'].extend(fragment_data[0])
            cluster_result['fragment_batch_ids'].extend(fragment_data[1])
            cluster_result['fragment_seg'].extend(fragment_data[2])

        # Format Fragments
        fragments_result = format_fragments(cluster_result['fragment_clusts'],
                                            cluster_result['fragment_batch_ids'],
                                            cluster_result['fragment_seg'],
                                            input[0][:, self.batch_col],
                                            batch_size=self.batch_size)

        cnn_result.update({'frag_dict':fragments_result})

        cnn_result.update({
            'fragment_clusts': fragments_result['fragment_clusts'],
            'fragment_seg': fragments_result['fragment_seg'],
            'fragment_batch_ids': fragments_result['fragment_batch_ids']
        })

        if self.enable_cnn_clust or self.enable_dbscan:
            cnn_result.update({'segment_label_tmp': [semantic_labels] })
            if label_clustering is not None:
                if 'input_rescaled' in cnn_result:
                    label_clustering[0][:, VALUE_COL] = input[0][:, VALUE_COL]
                cnn_result.update({'cluster_label_adapted': label_clustering })

        # if self.use_true_fragments and coords is not None:
        #     print('adding true points info')
        #     cnn_result['true_points'] = coords

        return cnn_result, input

    @staticmethod
    def get_extra_gnn_features(data, result, clusts, clusts_seg, classes,
            add_points=True, add_value=True, add_shape=True):
        """
        Extracting extra features to feed into the GNN particle aggregators

        Parameters
        ----------
        data : torch.Tensor
            Tensor of input voxels to the particle aggregator
        result : dict
            Dictionary of output of the CNN stages
        clusts : List[numpy.ndarray]
            List of clusters representing the fragment or particle objects
        clusts_seg : numpy.ndarray
            Array of cluster semantic types
        classes : List, optional
            List of semantic classes to include in the output set of particles
        add_points : bool, default True
            If `True`, add particle points as node features
        add_value : bool, default True
            If `True`, add mean and std voxel values as node features
        add_shape : bool, default True
            If `True`, add cluster semantic shape as a node feature

        Returns
        -------
        index : np.ndarray
            Index to select fragments belonging to one of the requested classes
        kwargs : dict
            Keys can include `points` (if `add_points` is `True`)
            and `extra_feats` (if `add_value` or `add_shape` is True).
        """
        # If needed, build a particle mask based on semantic classes
        if classes is not None:
            mask = np.zeros(len(clusts_seg), dtype=bool)
            for c in classes:
                mask |= (clusts_seg == c)
            index = np.where(mask)[0]
        else:
            index = np.arange(len(clusts))

        # Get the particle end points, if requested
        kwargs = {}
        if add_points:
            coords     = data[0][:, COORD_COLS].detach().cpu().numpy()
            ppn_points = result['ppn_points'][0].detach().cpu().numpy()
            points     = get_particle_points(coords, clusts[index],
                    clusts_seg[index], ppn_points)

            kwargs['points'] = torch.tensor(points,
                    dtype=torch.float, device=data[0].device)

        # Get the supplemental information, if requested
        if add_value or add_shape:
            extra_feats = torch.empty((len(index), 2*add_value + add_shape),
                    dtype=torch.float, device=data[0].device)
            if add_value:
                extra_feats[:,:2] = get_cluster_features_extended(data[0],
                        clusts[index], add_value=True, add_shape=False)
            if add_shape:
                extra_feats[:,-1] = torch.tensor(clusts_seg[index],
                        dtype=torch.float, device=data[0].device)

            kwargs['extra_feats'] = torch.tensor(extra_feats,
                    dtype=torch.float, device=data[0].device)

        return index, kwargs


class FullChainLoss(torch.nn.Module):
    """
    Loss function for the full chain.

    See Also
    --------
    FullChain
    """
    modes = FullChain.modes

    def __init__(self, chain, uresnet_deghost=None, uresnet_deghost_loss=None,
                 uresnet=None, uresnet_loss=None, uresnet_ppn=None,
                 uresnet_ppn_loss=None, graph_spice_loss=None,
                 grappa_shower_loss=None, grappa_track_loss=None,
                 grappa_particle_loss=None, grappa_inter_loss=None, **kwargs):
        """Initialize the full chain model.

        Parameters
        ----------
        chain : dict
            Dictionary of parameters used to configure the chain
        uresnet_deghost : dict, optional
            Deghosting model configuration
        uresnet_deghost_loss : dict, optional
            Deghosting loss configuration
        uresnet : dict, optional
            Segmentation model configuration
        uresnet_loss : dict, optional
            Segmentation loss configuration
        uresnet_ppn: dict, optional
            Segmentation and point proposal model configuration
        uresnet_ppn_loss : dict, optional
            Segmentation and point proposal loss configuration
        graph_spice_loss : dict, optional
            Supervised connected component clustering loss configuration
        grappa_shower : dict, optional
            Shower aggregation model configuration
        grappa_track : dict, optional
            Track aggregation model configuration
        grappa_particle : dict, optional
            Global particle aggregation configuration
        grappa_inter : dict, optional
            Interaction aggregation model configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the main chain configureation
        process_chain_config(self, **chain)

        # Initialize the deghosting loss
        if self.deghosting == 'uresnet':
            self.deghost_loss = SegmentationLoss(
                    uresnet_deghost, uresnet_deghost_loss)

        # Initialize the segmentation/PPN losses
        if self.segmentation == 'uresnet':
            assert ((uresnet_loss is not None) ^ 
                    (uresnet_ppn_loss is not None)), (
                    "If the segmentation is using UResNet, must provide the "
                    "`uresnet_loss` or `uresnet_ppn_loss` configuration block.")
            if uresnet_loss is not None:
                self.uresnet_loss = SegmentationLoss(uresnet, uresnet_loss)
            else:
                self.uresnet_ppn_loss = UResNetPPNLoss(
                        **uresnet_ppn, **uresnet_ppn_loss)

        # Initialize the graph-SPICE loss
        if 'graph_spice' in self.dense_clustering:
            self.graph_spice_loss = GraphSPICELoss(graph_spice_loss)

        # TODO Add the GrapPA losses

def process_chain_config(self, dump_config=False, **parameters):
    """Process the full chain configuration and dump it.

    Parameters
    ----------
    dump_config : bool, default False
        Whether to print out the chain configuration or not
    **parameters : dict
        Dictionary of chain configuration parameters
    """
    # Store the modes for each step of the reconstruction. Make sure that
    # that the configuration is recognized.
    for module, valid_modes in self.modes.items():
        valid_modes = [None, 'label'] + valid_modes
        assert module in parameters, (
                f"Must configure the {module} stage in the `chain` block. "
                f"The {module} mode should be one of {valid_modes}.")
        assert parameters[module] in valid_modes, (
                f"The {module} mode should be one of {valid_modes}.")
        setattr(self, module, parameters[module])

    # Do some logic checks on the chain parameters
    assert not self.charge_rescaling or self.deghosting, (
            "Charge rescaling is meaningless without deghosting")
    assert (not self.point_proposal == 'ppn' or
            self.segmentation == 'uresnet'), (
            "For PPN to work, need the UResNet segmentation backbone")
    assert (not self.shower_primary == 'gnn' or
            self.shower_aggregation == 'gnn'), (
            "To use GNN for shower primaries, must aggregate showers with it")
    assert (not self.particle_aggregation or
            (not self.shower_aggregation and not self.track_aggregation)), (
            "Use particle aggregator or shower/track aggregators, not both")

    if dump_config:
        logger.info(f"Full chain configuration:")
        for k, v in parameters.items():
            v = v if v is not None else "null"
            logger.info(f"  {k:<27}: {v}")
        logger.info("")
