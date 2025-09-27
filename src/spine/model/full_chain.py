"""Module with the core full reconstruction chain."""

from typing import List, Union

import numpy as np
import torch
import yaml
from torch_scatter import scatter_mean, scatter_std

from spine.data import IndexBatch, RunInfo, TensorBatch
from spine.utils.calib import CalibrationManager
from spine.utils.cluster.label import ClusterLabelAdapter
from spine.utils.ghost import ChargeRescaler
from spine.utils.globals import (
    CLUST_COL,
    COORD_COLS,
    DELTA_SHP,
    GHOST_SHP,
    MICHL_SHP,
    SHAPE_COL,
    SHOWR_SHP,
    TRACK_SHP,
    VALUE_COL,
)
from spine.utils.gnn.cluster import form_clusters_batch, get_cluster_label_batch
from spine.utils.gnn.evaluation import primary_assignment_batch
from spine.utils.logger import logger
from spine.utils.ppn import ParticlePointPredictor

from .graph_spice import GraphSPICE, GraphSPICELoss
from .grappa import GrapPA, GrapPALoss
from .layer.common.dbscan import DBSCAN
from .uresnet import SegmentationLoss, UResNetSegmentation
from .uresnet_ppn import UResNetPPN, UResNetPPNLoss

# from spine.model.layer.cnn.cnn_encoder import SparseResidualEncoder ??
# TODO: replace this with MultiParticleImageClassifier
# TODO: raname it something more generic like ParticleClusterImageClassifier?


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

    The `chain` section enables or disables specific stages of the full
    chain. When a module is disabled through this section, it will not be
    constructed. The configuration blocks for each enabled module should
    also live under the `modules` section of the configuration.
    """

    # TODO: update
    MODULES = [
        "grappa_shower",
        "grappa_track",
        "grappa_inter",
        "grappa_shower_loss",
        "grappa_track_loss",
        "grappa_inter_loss",
        "full_chain_loss",
        "graph_spice",
        "graph_spice_loss",
        "fragment_clustering",
        "chain",
        "dbscan_frag",
        ("uresnet_ppn", ["uresnet", "ppn"]),
    ]

    # Valid chain stage modes
    _modes = (
        ("deghosting", ("uresnet",)),
        ("charge_rescaling", ("collection", "average")),
        ("segmentation", ("uresnet",)),
        ("point_proposal", ("ppn",)),
        ("fragmentation", ("dbscan", "graph_spice", "dbscan_graph_spice")),
        ("shower_aggregation", ("skip", "grappa")),
        ("shower_primary", ("skip", "grappa")),
        ("track_aggregation", ("skip", "grappa")),
        ("particle_aggregation", ("skip", "grappa")),
        ("inter_aggregation", ("grappa",)),
        ("particle_identification", ("grappa", "image")),
        ("primary_identification", ("grappa",)),
        ("orientation_identification", ("grappa",)),
        ("calibration", ("apply",)),
    )

    # Valid calibration stages
    _calib_stages = (
        "segmentation",
        "fragmentation",
        "particle_aggregation",
        "inter_aggregation",
        "particle_identification",
    )

    def __init__(
        self,
        chain,
        uresnet_deghost=None,
        uresnet=None,
        uresnet_ppn=None,
        adapt_labels=None,
        predict_points=None,
        graph_spice=None,
        dbscan=None,
        grappa_shower=None,
        grappa_track=None,
        grappa_particle=None,
        grappa_inter=None,
        calibration=None,
        uresnet_deghost_loss=None,
        uresnet_loss=None,
        uresnet_ppn_loss=None,
        graph_spice_loss=None,
        grappa_shower_loss=None,
        grappa_track_loss=None,
        grappa_particle_loss=None,
        grappa_inter_loss=None,
    ):
        """Initialize the full chain model.

        Parameters
        ----------
        chain : dict
            Dictionary of parameters used to configure the chain
        uresnet_deghost : dict, optional
            Deghosting model configuration
        uresnet_ppn : dict, optional
            Segmentation and point proposal model configuration
        adapt_labels : dict, optional
            Parameters for the cluster label adaptation (if non-standard)
        predict_points : dict, optional
            Parameters for the particle point predictor (if non-standard)
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
            Calibration manager configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the main chain configureation
        process_chain_config(self, **chain, dump_config=True)

        # Initialize the deghosting model
        if self.deghosting is not None and self.deghosting == "uresnet":
            assert uresnet_deghost is not None, (
                "If the deghosting is using UResNet, must provide the "
                "`uresnet_deghost` configuration block."
            )
            self.uresnet_deghost = UResNetSegmentation(uresnet_deghost)

        # Initialize the charge rescaling process (adapt to ghost predictions)
        if self.charge_rescaling is not None:
            self.charge_rescaler = ChargeRescaler(
                collection_only=self.charge_rescaling == "collection"
            )

        # Initialize the semantic segmentation model (+ point proposal)
        if self.segmentation is not None and self.segmentation == "uresnet":
            assert (uresnet is not None) ^ (uresnet_ppn is not None), (
                "If the segmentation is using UResNet, must provide the "
                "`uresnet` or `uresnet_ppn` configuration block."
            )
            if uresnet is not None:
                self.uresnet = UResNetSegmentation(uresnet)
            else:
                self.uresnet_ppn = UResNetPPN(**uresnet_ppn)

        # Initialize the relabeling process (adapt to the semantic predictions)
        self.label_adapter = ClusterLabelAdapter(**(adapt_labels or {}))

        # Initialize the point predictor (for fragment/particle clusters)
        self.point_predictor = ParticlePointPredictor(**(predict_points or {}))

        # Initialize the dense clustering model
        self.fragment_shapes = []
        if self.fragmentation is not None and (
            "dbscan" in self.fragmentation or "graph_spice" in self.fragmentation
        ):
            if "dbscan" in self.fragmentation:
                assert dbscan is not None, (
                    "If the fragmentation is done using DBSCAN, must "
                    "provide the `dbscan` configuration block."
                )
                self.dbscan = DBSCAN(**dbscan)
                self.fragment_shapes.extend(self.dbscan.shapes)
            if "graph_spice" in self.fragmentation:
                assert graph_spice is not None, (
                    "If the fragmentation is done using Graph-SPICE, must "
                    "provide the `graph_spice` configuration block."
                )
                self.graph_spice = GraphSPICE(graph_spice)
                self.fragment_shapes.extend(self.graph_spice.shapes)

            # Check that there is no redundancy between the fragmenters
            uniques, counts = np.unique(self.fragment_shapes, return_counts=True)
            assert np.all(uniques == np.arange(4)), (
                "All four expected semantic classes should be fragmented "
                "by either DBSCAN or Graph-SPICE."
            )
            assert np.all(counts) == 1, (
                "Some of the shapes appear in both the DBSCAN and the "
                "Graph-SPICE based fragmentation. Ambiguous."
            )

        # Initialize the graph-based aggregator modules
        grappa_configs = {
            "shower": grappa_shower,
            "track": grappa_track,
            "particle": grappa_particle,
            "inter": grappa_inter,
        }
        for stage, config in grappa_configs.items():
            if getattr(self, f"{stage}_aggregation") == "grappa":
                name = f"grappa_{stage}"
                assert config is not None, (
                    f"If the {stage} aggregation is done using GrapPA, "
                    f"must provide the {name} configuration block."
                )
                setattr(self, name, GrapPA(config))
                assert (
                    getattr(self, name).make_groups == True
                ), "The aggregators should have `make_groups: true`"

        # Initialize the standalone particle classifier
        # TODO (likely an image classifier)

        # Initialize the interaction-classification module
        # TODO (could be done by either CNN or graph-level GNN)

        # Initialize the calibrator manager
        self.calibration_stage = None
        if self.calibration == "apply":
            assert calibration is not None, (
                "If the calibration is to be applied, must provide the "
                "`calibration` configuration block."
            )
            assert "stage" in calibration, (
                "If the calibration is to be applied, must provide the "
                "`stage` to specify where to apply it."
            )
            self.calibration_stage = calibration.pop("stage")
            self.calibrator = CalibrationManager(**calibration)
            calibration["stage"] = self.calibration_stage

    @property
    def modes(self):
        """Dictionary of (stage, modes) pairs which determine which options
        are available to each of the the reconstruction stage.

        Returns
        -------
        Dict[str, Tuple(str)]
            Dictionary of (stage, modes) pairs to be used
        """
        return dict(self._modes)

    def forward(
        self,
        data,
        sources=None,
        seg_label=None,
        clust_label=None,
        coord_label=None,
        energy_label=None,
        meta=None,
        run_info=None,
    ):
        """Run a batch of data through the full chain.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        sources : TensorBatch, optional
            (N, 2) tensor of module/tpc pair for each voxel
        seg_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        coord_label : TensorBatch, optional
            (N, 1 + D + N_p) Tensor of point of interest labels
            - N_p is the number point labels
        energy_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of true energy deposition values
            - 1 is the energy deposition value in each voxel
        meta : Meta, optional
            Image metadata information
        run_info : List[RunInfo], optional
            Object containing information about the run, subrun and event

        Returns
        -------
        TODO
        """
        # Initialize the full chain output dictionary
        self.result = {}

        # Run the deghosting step
        data, sources = self.run_deghosting(data, sources, seg_label, clust_label)

        # Run the semantic segmentation (and point proposal) stage
        if self.calibration_stage == "segmentation":
            data = self.run_calibration(data, sources, energy_label, meta, run_info)
        clust_label = self.run_segmentation_ppn(data, seg_label, clust_label)

        # Run the fragmentation stage
        if self.calibration_stage == "fragmentation":
            data = self.run_calibration(data, sources, energy_label, meta, run_info)
        self.run_fragmentation(data, clust_label)

        # Run the particle aggregation
        if self.calibration_stage == "particle_aggregation":
            data = self.run_calibration(data, sources, energy_label, meta, run_info)
        self.run_part_aggregation(data, clust_label, coord_label)

        # Run the interaction aggregation
        if self.calibration_stage == "inter_aggregation":
            data = self.run_calibration(data, sources, energy_label, meta, run_info)
        self.run_inter_aggregation(data, clust_label, coord_label)

        # Run an independant particle classification stage
        if self.calibration_stage == "particle_classification":
            data = self.run_calibration(data, sources, energy_label, meta, run_info)
        # TODO

        # Run the interaction classification
        # TODO

        # Return
        return self.result

    def run_deghosting(self, data, sources=None, seg_label=None, clust_label=None):
        """Run the deghosting algorithm.

        This removes points that are artifacts of the tomographic
        reconstruction. This is only relevant for detectors producing 2D
        projections of an event. If requested, charge is rescaled according
        to the deghosting mask.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        sources : TensorBatch, optional
            (N, 2) tensor of module/tpc pair for each voxel
        seg_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of segmentation labels
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels

        Returns
        -------
        TensorBatch
            (N, 1 + D + N_f) tensor of deghosted voxel/value pairs
        """
        if self.deghosting == "uresnet":
            # Pass the data through the model
            res_deghost = self.uresnet_deghost(data)

            # Store the ghost scores and the ghost mask
            ghost_tensor = res_deghost["segmentation"].tensor
            ghost_pred = torch.argmax(ghost_tensor, dim=1)
            data_adapt = TensorBatch(
                data.tensor[ghost_pred == 0],
                batch_size=data.batch_size,
                has_batch_col=True,
                coord_cols=data.coord_cols,
            )
            ghost_pred = TensorBatch(ghost_pred, data.counts)

            # Rescale the charge, if requested
            if self.charge_rescaling is not None:
                charges = self.charge_rescaler(data_adapt)
                tensor_deghost = data_adapt.tensor[:, :-6]
                tensor_deghost[:, VALUE_COL] = charges
                data_adapt.data = tensor_deghost

            self.result["ghost"] = res_deghost["segmentation"]
            self.result["ghost_pred"] = ghost_pred
            self.result["data_adapt"] = data_adapt

            # If sources are provided, narrow them down to non-ghosts
            sources_adapt = None
            if sources is not None:
                # Store the sources of the predicted non-ghosts
                sources_adapt = TensorBatch(
                    sources.tensor[ghost_pred.tensor == 0], data_adapt.counts
                )
                self.result["sources_adapt"] = sources_adapt

                # Store the sources of the true non-ghosts
                if seg_label is not None:
                    ghost_label = seg_label.tensor[:, SHAPE_COL] < GHOST_SHP
                    if clust_label is not None:
                        counts = clust_label.counts
                    else:
                        counts = TensorBatch(
                            seg_label.tensor[ghost_label],
                            batch_size=seg_label.batch_size,
                            has_batch_col=True,
                        ).counts

                    sources_label = TensorBatch(sources.tensor[ghost_label], counts)
                    self.result["sources_label"] = sources_label

            return data_adapt, sources_adapt

        elif self.deghosting == "label":
            # Use ghost labels to remove ghost voxels from the input
            assert seg_label is not None, "Must provide `seg_label` to deghost with it."
            ghost_pred = (seg_label.tensor[:, SHAPE_COL] == GHOST_SHP).long
            tensor_deghost = data.tensor[ghost_pred == 0]

            # Use the label rescaled charge, if requested
            if self.charge_rescaling is not None:
                assert clust_label is not None, (
                    "Must provide `clust_label` to fetch the true " "rescaled charge."
                )
                tensor_deghost[:, VALUE_COL] = clust_label[:, VALUE_COL]

            # Store and return
            ghost_pred = TensorBatch(ghost_pred, data.counts)
            data_adapt = TensorBatch(
                tensor_deghost, batch_size=data.batch_size, coord_cols=data.coord_cols
            )
            self.result["ghost_pred"] = ghost_pred
            self.result["data_adapt"] = data_adapt

            # If sources are provided, narrow them down to non-ghosts
            sources_adapt = None
            if sources is not None:
                sources_adapt = TensorBatch(
                    sources.tensor[ghost_pred.tensor == 0], data_adapt.counts
                )
                self.result["sources_adapt"] = sources_adapt
                if seg_label is not None:
                    self.result["sources_label"] = sources_adapt

            return data_adapt, sources_adapt

        else:
            # If sources are provided, set the expected output
            sources_adapt = None
            if sources is not None:
                self.result["sources_adapt"] = sources
                if seg_label is not None:
                    self.result["sources_label"] = sources

            # Nothing to do
            return data, sources_adapt

    def run_segmentation_ppn(self, data, seg_label=None, clust_label=None):
        """Run the semantic segmentation and the point proposal algorithms.

        This classifies each individual voxel in the image into different
        particle topological categories and identifies poins of interest,
        namely track end points and shower fragment start points.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        seg_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of segmentation labels
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
        """
        if self.segmentation == "uresnet":
            # Run the data through the appropriate model
            if hasattr(self, "uresnet"):
                res_seg = self.uresnet(data)
            else:
                res_seg = self.uresnet_ppn(data)

            # If the deghosting is done as part of this step, process it
            if "ghost" in res_seg:
                # Store the ghost scores and the ghost mask
                ghost_tensor = res_deghost["host"].tensor
                ghost_pred = torch.argmax(ghost_tensor, dim=1)
                data_adapt = TensorBatch(
                    data.tensor[ghost_pred == 0],
                    batch_size=data.batch_size,
                    coord_cols=data.coord_cols,
                )
                ghost_pred = TensorBatch(ghost_pred, data.counts)

                self.result["ghost_pred"] = ghost_pred
                self.result["data_adapt"] = data_adapt

                # If there are PPN outputs, deghost them
                if "ppn_points" in res_seg:
                    deghost_index = torch.where(ghost_pred == 0)[0]
                    res_seg["ppn_points"] = res_seg["ppn_points"][deghost_index]
                    for key in [
                        "ppn_masks",
                        "ppn_coords",
                        "ppn_layers",
                        "ppn_classify_endpoints",
                    ]:
                        if key in res_seg:
                            res_seg[key][-1] = res_seg[key][-1][deghost_index]

            # Update the result dictionary
            self.result.update(res_seg)
            seg_pred = torch.argmax(res_seg["segmentation"].tensor, dim=1)

            self.result["seg_pred"] = TensorBatch(seg_pred, data.counts)

            # If the rest of the chain is run, must adapt cluster labels now
            if seg_label is not None and clust_label is not None:
                seg_pred = self.result["seg_pred"]
                ghost_pred = self.result.get("ghost_pred", None)
                clust_label = self.label_adapter(
                    clust_label, seg_label, seg_pred, ghost_pred
                )

                self.result["clust_label_adapt"] = clust_label

        elif self.segmentation == "label":
            # Use the segmentation labels
            assert seg_label is not None, "Must provide `seg_label` to segment with it."
            seg_pred = seg_label.tensor[:, SHAPE_COL]

            self.result["seg_pred"] = TensorBatch(seg_pred, data.counts)

        return clust_label

    def run_fragmentation(self, data, clust_label=None):
        """Run the fragmentation algorithm, i.e. the dense clustering step.

        This breaks down each topological class individually into a set of
        fragments, each composed of contiguous voxels which belong to a single
        particle instance, but do not necessarily make up the whole instance.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
        """
        # Nothing to do if there is no fragmentation requested
        if self.fragmentation is None:
            return

        # Initialize the fragment-level output
        counts = np.zeros(data.batch_size, dtype=np.int64)
        fragments = IndexBatch([], data.edges[:-1], counts, [])
        fragment_shapes = TensorBatch(np.empty(0, dtype=np.int64), counts)

        # Append the fragment list
        if "dbscan" in self.fragmentation:
            # Run DBSCAN
            clusts, clust_shapes = self.dbscan(data, **self.result)

            # Append
            fragments = fragments.merge(clusts)
            fragment_shapes = fragment_shapes.merge(clust_shapes)

        if "graph_spice" in self.fragmentation:
            # Run Graph-SPICE
            seg_pred = TensorBatch(self.result["seg_pred"].tensor[:, None], data.counts)
            res_gs = self.graph_spice(data, seg_pred, clust_label)

            # Update the global result with the graph_spice output
            self.result.update(
                {f"graph_spice_{key}": val for key, val in res_gs.items()}
            )

            # If fragments are produced, store them
            if "clusts" in res_gs:
                # Fragments point at a partial tensor, adapt them
                clusts = res_gs["clusts"]
                clust_shapes = res_gs["clust_shapes"]
                filter_index = res_gs["filter_index"].index
                for i, f in enumerate(clusts.index_list):
                    clusts.data[i] = filter_index[f]

                clusts.offsets = data.edges[:-1]

                # Append
                fragments = fragments.merge(clusts.to_numpy())
                fragment_shapes = fragment_shapes.merge(clust_shapes)

        if self.fragmentation == "label":
            assert (
                clust_label is not None
            ), "Must provide `clust_label` to use it for fragmentation."
            fragments = form_clusters_batch(clust_label.to_numpy(), column=CLUST_COL)
            fragment_shapes = get_cluster_label_batch(clust_label, fragments)

        if fragments is not None:
            self.result["fragment_clusts"] = fragments
            self.result["fragment_shapes"] = fragment_shapes

    def run_part_aggregation(self, data, clust_label=None, coord_label=None):
        """Run the particle aggreation step.

        This step gathers particle fragments into complete particle instances.
        It either aggregates shower and track fragments independently or
        jointly into a single step.

        In the process of shower aggregation, shower primaries can
        be identified.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
        coord_label : TensorBatch, optional
            (N, 1 + D + 6) Array of label particle end points
        """
        # Fetch the list of fragments and semantic classes
        if "fragment_clusts" not in self.result:
            return

        fragments = self.result["fragment_clusts"]
        fragment_shapes = self.result["fragment_shapes"]

        # If GrapPA is run on track/shower separately, prepare merged products
        merge = {}
        if self.shower_aggregation == "grappa" or self.track_aggregation == "grappa":
            num_clusts = len(fragments.index_list)
            kwargs = {"dtype": data.dtype, "device": data.device}
            merge["start_points"] = torch.full((num_clusts, 3), -np.inf, **kwargs)
            merge["end_points"] = torch.full((num_clusts, 3), -np.inf, **kwargs)
            merge["node_pred"] = torch.full((num_clusts, 2), -np.inf, **kwargs)
            merge["group_pred"] = -np.ones(num_clusts, dtype=np.int64)

        # Initialize the particle-level output
        counts = np.zeros(fragments.batch_size, dtype=np.int64)
        particles = IndexBatch([], fragments.offsets, counts, [])
        particle_shapes = TensorBatch(np.empty(0, dtype=np.int64), counts)
        particle_primaries = IndexBatch([], fragments.offsets, counts, [])

        # Loop over GraPA models, append the particle list
        shapes = {
            "shower": [SHOWR_SHP, MICHL_SHP, DELTA_SHP],
            "track": [TRACK_SHP],
            "particle": self.fragment_shapes,
        }
        use_primary = {"shower": True, "track": False, "particle": True}
        for name in ["shower", "track", "particle"]:
            # Dispatch the input to the right aggregation method
            switch = getattr(self, f"{name}_aggregation")

            if switch == "grappa":
                # Use GraPA to aggregate instances
                prefix = f"{name}_fragment" if name != "particle" else "fragment"
                model = getattr(self, f"grappa_{name}")
                (groups, group_shapes, group_primaries, shape_index) = self.run_grappa(
                    prefix,
                    model,
                    data,
                    fragments,
                    fragment_shapes,
                    coord_label,
                    aggregate_shapes=True,
                    shape_use_primary=use_primary[name],
                    retain_primaries=use_primary[name],
                )

                if merge and shape_index is not None:
                    for key in merge:
                        if key != "group_pred":
                            if key != "node_pred" or use_primary[name]:
                                merge[key][shape_index] = self.result[
                                    f"{prefix}_{key}"
                                ].tensor
                        else:
                            max_id = np.max(merge[key]) + 1
                            merge[key][shape_index] = (
                                self.result[f"{prefix}_{key}"].tensor + max_id
                            )

            elif switch == "label":
                # Use cluster labels to aggregate instances
                (groups, group_shapes, group_primaries, shape_index) = (
                    self.group_labels(
                        shapes[name],
                        data,
                        fragments,
                        fragment_shapes,
                        aggregate_shapes=True,
                        shape_use_primary=use_primary[name],
                        retain_primaries=use_primary[name],
                    )
                )

            elif switch == "skip":
                # Leave the shower fragments as is
                groups, group_shapes, shape_index = self.restrict_clusts(
                    fragments, fragment_shapes, shapes[name]
                )
                group_primaries = groups

            else:
                # Skip if there nothing to do
                continue

            # Append
            particles = particles.merge(groups)
            particle_shapes = particle_shapes.merge(group_shapes)
            particle_primaries = particle_primaries.merge(group_primaries)

        # Store merged fragment properties, if needed
        if merge:
            for key, value in merge.items():
                # Convert to a TensorBatch
                coord_cols = np.array([0, 1, 2]) if key.endswith("points") else None
                self.result[f"fragment_{key}"] = TensorBatch(
                    value, counts=fragments.counts, coord_cols=coord_cols
                )

                # For group predictions, reorganize them by batch entry
                if key == "group_pred":
                    group_pred = self.result[f"fragment_{key}"]
                    offset = 0
                    for b, group_pred_b in enumerate(group_pred.split()):
                        lower, upper = group_pred.edges[b], group_pred.edges[b + 1]
                        group_pred_b = np.unique(group_pred_b, return_inverse=True)[-1]
                        group_pred.data[lower:upper] = offset + group_pred_b
                        offset += len(group_pred_b)

        # Store particle objects
        self.result["particle_clusts"] = particles
        self.result["particle_shapes"] = particle_shapes
        self.result["particle_primaries"] = particle_primaries

    def run_inter_aggregation(self, data, clust_label=None, coord_label=None):
        """Run the interaction aggreation step.

        This step gathers particles into complete interaction instances.

        In the process of interaction aggregation, other tasks may be performed:
        - Particle identification
        - Primary tagging (particle coming from the interaction vertex)
        - Orientation tagging (order start/end points of particles)
        - Vertex reconstruction

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
        coord_label : TensorBatch, optional
            (N, 1 + D + 6) Array of label particle end points
        """
        # Fetch the list of particles and semantic classes
        if "particle_clusts" not in self.result:
            return

        particles = self.result["particle_clusts"]
        particle_shapes = self.result["particle_shapes"]
        particle_primaries = self.result["particle_primaries"]

        # Append the interaction list
        if self.inter_aggregation == "grappa":
            # Use GraPA to aggregate instances
            interactions, _, _, _ = self.run_grappa(
                "particle",
                self.grappa_inter,
                data,
                particles,
                particle_shapes,
                particle_primaries,
                coord_label,
                point_use_primary=True,
            )

        elif self.inter_aggregation == "label":
            # Use cluster labels to aggregate instances
            interactions, _, _, _ = self.group_labels(
                shapes[name], data, particles, particle_shapes
            )

        # Store interaction objects
        if self.inter_aggregation is not None:
            self.result["interaction_clusts"] = interactions

    def run_calibration(
        self, data, sources=None, energy_label=None, meta=None, run_info=None
    ):
        """Run the calibration algorithm.

        This converts the raw charge values in ADC to energy depositions
        expressed in MeV. It applies gain, recombination, transparency
        and electron lifetime corrections.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        sources : TensorBatch, optional
            (N, 2) tensor of module/tpc pair for each voxel
        energy_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of true energy deposition values
            - 1 is the energy deposition value in each voxel
        meta : Meta, optional
            Image metadata information
        run_info : List[RunInfo], optional
            Object containing information about the run, subrun and event

        Returns
        -------
        TensorBatch
            (N, 1 + D + N_f) tensor of calibrated voxel/value pairs
        """
        if self.calibration == "apply":
            # Check that the metadata is provided
            assert meta is not None, (
                "Must provide the metadata to convert pixel coordinates "
                "to detector coordinates and apply calibrations."
            )

            # Loop over entries in the batch (might have different meta/run IDs)
            data_np = data.to_numpy().tensor
            sources = sources.to_numpy().tensor if sources is not None else None
            rep = data.batch_size // len(meta)
            for b in range(data.batch_size):
                # Fetch necessary information for this batch entry
                lower, upper = data.edges[b], data.edges[b + 1]
                data_b = data_np[lower:upper]
                voxels_b = data_b[:, COORD_COLS]
                values_b = data_b[:, VALUE_COL]
                sources_b = sources[lower:upper] if sources is not None else None

                # Fetch meta/run information for this batch entry
                meta_b = meta[b // rep]
                run_id = run_info[b // rep].run if run_info is not None else None

                # Calibrate voxel values
                values_b = self.calibrator(
                    voxels_b,
                    values_b,
                    sources_b,
                    run_id,
                    meta=meta_b,
                    module_id=b % rep,
                )

                data.tensor[lower:upper, VALUE_COL] = torch.tensor(
                    values_b, dtype=data.dtype, device=data.device
                )

            self.result["data_adapt"] = data

        elif self.calibration == "label":
            # Use energy labels to give values to each voxel
            assert (
                energy_label is not None
            ), "Must provide `energy_label` to do label-based calibration."
            data.tensor[:, VALUE_COL] = energy_label.tensor[:, VALUE_COL]

            self.result["data_adapt"] = data

        return data

    def run_grappa(
        self,
        prefix,
        model,
        data,
        clusts,
        clust_shapes,
        clust_primaries=None,
        coord_label=None,
        aggregate_shapes=False,
        shape_use_primary=False,
        point_use_primary=False,
        retain_primaries=False,
    ):
        """Run the GNN-based particle aggregator.

        Parameters
        ----------
        prefix : str
            Name of the aggregation step
        model : GraPA
            GraPA model to execute for this aggregation step
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clusts : IndexBatch
            List of clusters to aggregate using GrapPA
        clust_shapes : TensorBatch
            Semantic type of each of the clusters
        clust_primaries : IndexBatch
            List of primary fragments associated with each input cluster
        coord_label : TensorBatch, optional
            (N, 1 + D + 6) Array of label particle end points
        aggregate_shapes : bool, default False
            Combine shapes to give a shape to the aggregated object
        shape_use_primary : bool, default False
            Use primary shape as the group shape
        point_use_primary : bool, default False
            Use the primary fragment to get the points
        retain_primaries : bool, default False
            Retain the primary cluster in the aggregated group

        Returns
        -------
        groups : IndexBatch
            List of cluster groups aggregated using GrapPA
        group_shapes : TensorBatch
            Semantic type of each of the cluster groups
        group_primaries : IndexBatch
            List of primary clusters for each group
        shape_index : np.ndarray
            List of indexes used to restrict the original cluster list
        """
        # Restrict the clusters to those in the input of the model
        clusts, clust_shapes, shape_index = self.restrict_clusts(
            clusts, clust_shapes, model.node_type
        )

        # Prepare the input to the aggregation stage
        grappa_input = self.prepare_grappa_input(
            model,
            data,
            clusts,
            clust_shapes,
            clust_primaries,
            coord_label,
            point_use_primary,
        )

        # Pass it through GrapPA, produce shower instances
        res_grappa = model(**grappa_input)
        self.result.update({f"{prefix}_{k}": v for k, v in res_grappa.items()})

        # If requested, convert the node predictions to a primary mask
        group_pred = res_grappa["group_pred"]
        primary_mask = None
        if shape_use_primary or retain_primaries:
            assert "node_pred" in res_grappa, (
                "Must provide node predictions to use primary shape "
                "or preserve the list of primary clusters."
            )
            node_pred = res_grappa["node_pred"].to_numpy()
            primary_mask = primary_assignment_batch(node_pred, group_pred)

        # Build shower instances, get their semantic type
        return (
            *self.build_groups(
                clusts,
                clust_shapes,
                group_pred,
                primary_mask,
                aggregate_shapes,
                shape_use_primary,
                retain_primaries,
            ),
            shape_index,
        )

    def group_labels(
        self,
        data,
        clusts,
        clust_shapes,
        aggregate_shapes=False,
        shape_use_primary=False,
        retain_primaries=False,
    ):
        """Aggregate particles using labels.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clusts : IndexBatch
            List of clusters to aggregate using GrapPA
        clust_shapes : TensorBatch
            Semantic type of each of the clusters
        aggregate_shapes : bool, default False
            Combine shapes to give a shape to the aggregated object
        shape_use_primary : bool, default False
            Use primary shape as the group shape
        retain_primaries : bool, default False
            Retain the primary cluster

        Returns
        -------
        groups : IndexBatch
            List of cluster groups aggregated using labels
        group_shapes : TensorBatch
            Semantic type of each of the cluster groups
        group_primaries : IndexBatch
            List of primary clusters for each group
        shape_index : np.ndarray
            List of indexes used to restrict the original cluster list
        """
        # Restrict the clusters to those in the input of the model
        clusts, clust_shapes, shape_index = self.restrict_clusts(
            clusts, clust_shapes, model.node_type
        )

        # If requested, convert the node predictions to a primary mask
        group_ids = get_cluster_label_batch(data, clusts, GROUP_COL)
        primary_mask = None
        if shape_use_primary:
            primary_mask = get_cluster_label_batch(data, clusts, PRGRP_COL)
            primary_mask = primary_mask.astype(bool)

        # Build shower instances, get their semantic type
        return (
            *self.build_groups(
                clusts,
                clust_shapes,
                group_ids,
                primary_mask,
                aggregate_shapes,
                shape_use_primary,
                retain_primaries,
            ),
            shape_index,
        )

    def restrict_clusts(self, clusts, clust_shapes, shapes):
        """Restricts a cluster list against a list of shapes.

        Parameters
        ----------
        clusts : IndexBatch
            List of clusters to aggregate using GrapPA
        clust_shapes : TensorBatch
            Semantic type of each of the clusters
        shapes : List[int]
            List of semantic shapes to restrict to

        Returns
        -------
        clusts : IndexBatch
            Restricted list of clusters
        clust_shapes : TensorBatch
            Restricted list of semantic types
        shape_index : np.ndarray
            List of indexes used to restrict the original cluster list
        """
        # Restrict the clusters to those in the input of the model
        shape_index = False
        if shapes != self.fragment_shapes:
            mask = np.zeros(len(clust_shapes.tensor), dtype=bool)
            for c in shapes:
                mask |= clust_shapes.tensor == c
            shape_index = np.where(mask)[0]

            batch_ids = clusts.batch_ids[shape_index]
            clusts_np = np.empty(len(clusts.index_list), dtype=object)
            clusts_np[:] = clusts.index_list
            clusts = IndexBatch(
                clusts_np[shape_index],
                offsets=clusts.offsets,
                single_counts=clusts.single_counts[shape_index],
                batch_ids=batch_ids,
                batch_size=clusts.batch_size,
            )
            clust_shapes = TensorBatch(clust_shapes.tensor[shape_index], clusts.counts)

        return clusts, clust_shapes, shape_index

    def prepare_grappa_input(
        self,
        model,
        data,
        clusts,
        clust_shapes,
        clust_primaries=None,
        coord_label=None,
        point_use_primaries=False,
    ):
        """Prepares the input to a GrapPA model.

        It builds the following input to GrpPA:
        - points: end points of fragments/particles
        - value: mean/std of charge distribution in each cluster
        - shape: shape of each fragment/particle

        Parameters
        ----------
        model : torch.nn.Module
            GrapPA model to feed information to
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clusts : IndexBatch
            List of clusters to aggregate using GrapPA
        clust_shapes : TensorBatch
            Semantic type of each of the clusters
        clust_primaries : IndexBatch, optional
            List of primary fragment within each cluster to aggregate
        coord_label : TensorBatch, optional
            (N, 1 + D + 6) Array of label particle end points
        point_use_primaries:
            Use the primary fragment only to infer primaries

        Returns
        -------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        clusts : IndexBatch
            Input clusters to the model
        shapes : TensorBatch
            List of semantic type of each clusters
        points : TensorBatch
            List of start/end points associated with each cluster
        extra : TensorBatch
            List of additional features to pass to the GrapPA model
        """
        # Store the basics
        grappa_input = {}
        grappa_input["data"] = data
        grappa_input["clusts"] = clusts
        grappa_input["shapes"] = clust_shapes
        if coord_label is not None:
            grappa_input["coord_label"] = coord_label

        # Get the particle end points, if requested
        if hasattr(model.node_encoder, "add_points") and model.node_encoder.add_points:
            # Fetch the cluster list to use to get points
            ref_clusts = clusts
            if point_use_primaries:
                assert (
                    clust_primaries is not None
                ), "If using primaries to get points, must provide them."
                ref_clusts = clust_primaries

            # Get and store the points
            points = self.point_predictor(
                data, ref_clusts, clust_shapes, self.result["ppn_points"]
            )

            grappa_input["points"] = points

        # Get the supplemental information, if requested
        if hasattr(model.node_encoder, "add_value") and (
            model.node_encoder.add_value or model.node_envoder.add_shape
        ):
            extra = []
            if model.node_encoder.add_value:
                values = data.tensor[clusts.full_index, VALUE_COL]
                index_ids = torch.tensor(
                    clusts.index_ids, dtype=torch.long, device=data.device
                )
                extra.append(scatter_mean(values, index_ids))
                extra.append(scatter_std(values, index_ids))

            if model.node_encoder.add_shape:
                shapes = torch.tensor(
                    clust_shapes.tensor, dtype=data.dtype, device=data.device
                )
                extra.append(shapes)

            grappa_input["extra"] = TensorBatch(torch.stack(extra).t(), clusts.counts)

        return grappa_input

    def build_groups(
        self,
        clusts,
        clust_shapes,
        group_pred,
        primary_mask=None,
        aggregate_shapes=False,
        shape_use_primary=False,
        retain_primaries=False,
    ):
        """Use groups predictions from GrapPA to build superstructures.

        Parameters
        ----------
        clusts : IndexBatch
            List of clusters to aggregate using GrapPA
        clust_shapes : TensorBatch
            Semantic type of each of the clusters
        group_pred : TensorBatch
            Group ID of each node in the GraPA output
        primary_mask : TensorBatch
            Binary mask as to whether a node is a group primary or not
        aggregate_shapes : bool, default False
            Combine shapes to give a shape to the aggregated object
        shape_use_primary : bool, default False
            Use primary shape as the group shape
        retain_primaries : bool, default False
            Retain the primary cluster in the aggregated group

        Returns
        -------
        groups : IndexBatch
            List of cluster groups aggregated using GrapPA
        group_shapes : TensorBatch
            Semantic type of each of the cluster groups
        group_primaries : IndexBatch
            List of primary clusters for each group
        """
        # Cast node_pred to numpy if it is provided
        groups, group_shapes, group_primaries = [], [], []
        counts, single_counts, single_primary_counts = [], [], []
        for b in range(group_pred.batch_size):
            # Fetch the subset of data for this batch
            clusts_b = clusts[b]
            offset_b = clusts.offsets[b]
            group_pred_b = group_pred[b]
            if aggregate_shapes:
                clust_shapes_b = clust_shapes[b]
            if primary_mask is not None:
                primary_mask_b = primary_mask[b]

            # Loop over unique group IDs
            group_ids = np.unique(group_pred_b)
            counts.append(len(group_ids))
            for g in group_ids:
                # Build the set of groups made up of input clusters
                group_index = np.where(group_pred_b == g)[0]
                groups.append(offset_b + np.concatenate(clusts_b[group_index]))
                single_counts.append(len(groups[-1]))

                # Extract the shape and primary ID for this group
                if primary_mask is not None:
                    primary_id = group_index[primary_mask_b[group_index]][0]
                    if aggregate_shapes:
                        group_shapes.append(clust_shapes_b[primary_id])
                    if retain_primaries:
                        group_primaries.append(offset_b + clusts_b[primary_id])
                        single_primary_counts.append(len(group_primaries[-1]))

                elif aggregate_shapes:
                    shapes, shape_counts = np.unique(
                        clust_shapes_b[group_index], return_counts=True
                    )
                    group_shapes.append(shapes[np.argmax(shape_counts)])

        groups = IndexBatch(groups, clusts.offsets, counts, single_counts)
        if aggregate_shapes:
            group_shapes = np.array(group_shapes, dtype=np.int64)
            group_shapes = TensorBatch(group_shapes, counts)
        if retain_primaries:
            group_primaries = IndexBatch(
                group_primaries, clusts.offsets, counts, single_primary_counts
            )
        else:
            group_primaries = groups

        return groups, group_shapes, group_primaries


class FullChainLoss(torch.nn.Module):
    """Loss function for the full chain.

    See Also
    --------
    FullChain
    """

    # Valid chain stage modes
    _modes = FullChain._modes

    def __init__(
        self,
        chain,
        uresnet_deghost=None,
        uresnet_deghost_loss=None,
        uresnet=None,
        uresnet_loss=None,
        uresnet_ppn=None,
        uresnet_ppn_loss=None,
        graph_spice=None,
        graph_spice_loss=None,
        grappa_shower_loss=None,
        grappa_track_loss=None,
        grappa_particle_loss=None,
        grappa_inter_loss=None,
        **kwargs,
    ):
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
        graph_spice : dict, optional
            Supervised connected component clustering model configuration
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
        if self.deghosting == "uresnet" and uresnet_deghost_loss is not None:
            self.deghost_loss = SegmentationLoss(uresnet_deghost, uresnet_deghost_loss)

        # Initialize the segmentation/PPN losses
        if self.segmentation == "uresnet":
            assert not (
                (uresnet_loss is not None) and (uresnet_ppn_loss is not None)
            ), (
                "If the segmentation is using UResNet, can provide either "
                "`uresnet_loss` or `uresnet_ppn_loss` configuration block."
            )
            if uresnet_loss is not None:
                self.uresnet_loss = SegmentationLoss(uresnet, uresnet_loss)
            elif uresnet_ppn_loss is not None:
                self.uresnet_ppn_loss = UResNetPPNLoss(
                    **uresnet_ppn, **uresnet_ppn_loss
                )

        # Initialize the graph-SPICE loss
        if (
            self.fragmentation is not None
            and "graph_spice" in self.fragmentation
            and graph_spice_loss is not None
        ):
            self.graph_spice_loss = GraphSPICELoss(graph_spice, graph_spice_loss)

        # Initialize the GraPA lossses
        self.grappa_losses = {
            "shower": grappa_shower_loss,
            "track": grappa_track_loss,
            "particle": grappa_particle_loss,
            "inter": grappa_inter_loss,
        }
        for stage, config in self.grappa_losses.items():
            if getattr(self, f"{stage}_aggregation") == "grappa" and config is not None:
                name = f"grappa_{stage}_loss"
                setattr(self, name, GrapPALoss(config))

    @property
    def modes(self):
        """Dictionary of (stage, modes) pairs which determine which options
        are available to each of the the reconstruction stage.

        Returns
        -------
        Dict[str, Tuple(str)]
            Dictionary of (stage, modes) pairs to be used
        """
        return dict(self._modes)

    def forward(
        self,
        seg_label=None,
        ppn_label=None,
        clust_label=None,
        clust_label_adapt=None,
        coord_label=None,
        graph_label=None,
        meta=None,
        ghost=None,
        ghost_pred=None,
        segmentation=None,
        seg_pred=None,
        **output,
    ):
        """Run the full chain output through the full chain loss.

        Parameters
        ----------
        seg_label : TensorBatch, optional
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        ppn_label : TensorBatch, optional
            (N, 1 + D + N_l) Tensor of PPN labels
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        clust_label_adapt : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels adapted to seg predictions
            - N_c is is the number of cluster labels
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Tensor of start/end point labels for each
            true particle in the image
        graph_label : EdgeIndexTensor, optional
            (2, E) Tensor of edges that correspond to physical
            connections between true particle in the image
        meta : Meta, optional
            Image metadata information
        ghost : TensorBatch, optional
            (N, 2) Tensor of logits from the deghosting model
        ghost_pred : TensorBatch, optional
            (N,) Tensor of ghost predictions
        segmentation : TensorBatch, optional
            (N, N_c) Tensor of logits from the segmentation model
        seg_pred : TensorBatch, optional
            (N) Semantic prediction for each point
        **output : dict, optional
            Additional outputs of the reconstruction chain
        """
        # Initialize the loss output
        self.result = {"accuracy": 1.0, "loss": 0.0, "num_losses": 0}

        # Apply the deghosting loss
        if self.deghosting == "uresnet" and hasattr(self, "deghost_loss"):
            # Convert segmentation labels to ghost labels
            ghost_label_tensor = seg_label.tensor.clone()
            ghost_label_tensor[:, SHAPE_COL] = (
                seg_label.tensor[:, SHAPE_COL] == GHOST_SHP
            )
            ghost_label = TensorBatch(ghost_label_tensor, seg_label.counts)

            # Store the loss dictionary
            res_deghost = self.deghost_loss(seg_label=ghost_label, segmentation=ghost)
            self.update_result(res_deghost, "ghost")

            # Restrict the segmentation labels and segmentation outputs
            # to true non-ghosts (do not apply deghosting loss twice)
            if segmentation is not None:
                # Find the index of true non-ghosts in the pred non-ghosts
                deghost_index = ghost_pred.tensor == 0
                seg_label_t = seg_label.tensor[deghost_index]
                index = seg_label_t[:, SHAPE_COL] < GHOST_SHP

                seg_label = TensorBatch(
                    seg_label_t[index],
                    batch_size=seg_label.batch_size,
                    has_batch_col=True,
                )
                segmentation = TensorBatch(segmentation.tensor[index], seg_label.counts)

        # Apply the segmentation and point proposal loss
        if self.segmentation == "uresnet":
            # Adapt the cluster labels to those corresponding to the
            # reconstructed semantic segmentation of the image
            clust_label = clust_label_adapt

            # Store the loss dictionary, if requested
            if hasattr(self, "uresnet_loss"):
                res_seg = self.uresnet_loss(
                    seg_label=seg_label, segmentation=segmentation
                )
                self.update_result(res_seg, "uresnet")

            elif hasattr(self, "uresnet_ppn_loss"):
                res_seg = self.uresnet_ppn_loss(
                    seg_label=seg_label,
                    ppn_label=ppn_label,
                    clust_label=clust_label,
                    segmentation=segmentation,
                    **output,
                )
                self.update_result(res_seg)

        # Apply the Graph-SPICE loss
        if (
            self.fragmentation is not None
            and "graph_spice" in self.fragmentation
            and hasattr(self, "graph_spice_loss")
        ):
            # Prepare Graph-SPICE loss input
            loss_dict = {}
            for key, value in output.items():
                if key.startswith("graph_spice_"):
                    loss_dict[key.replace("graph_spice_", "")] = value

            # Store the loss dictionary
            seg_pred = TensorBatch(seg_pred.tensor[:, None], clust_label.counts)
            res_gs = self.graph_spice_loss(
                seg_label=seg_pred, clust_label=clust_label, **loss_dict
            )
            self.update_result(res_gs, "graph_spice")

        # Apply the aggregation losses
        for stage in self.grappa_losses.keys():
            name = f"grappa_{stage}_loss"
            if getattr(self, f"{stage}_aggregation") == "grappa" and hasattr(
                self, name
            ):
                # Prepare the input to the loss function
                prefix = f"{stage}_fragment" if stage != "inter" else "particle"
                loss_dict = {}
                for k, v in output.items():
                    if f"{prefix}_" in k:
                        loss_dict[k.split(f"{prefix}_")[-1]] = v

                # Store the loss dictionary
                res_grappa = getattr(self, name)(
                    clust_label=clust_label,
                    coord_label=coord_label,
                    graph_label=graph_label,
                    meta=meta,
                    **loss_dict,
                )
                self.update_result(res_grappa, f"grappa_{stage}")

        return self.result

    def update_result(self, result, prefix=None):
        """Update loss and accuracy using the output of one of the module.

        Parameters
        ----------
        result : dict
            Dictionary output of the module
        prefix : str, optional
            Prefix to preface the loss output keys with
        """
        # Update the loss, accuracy and count
        self.result["loss"] += result["loss"]
        self.result["accuracy"] = (
            self.result["accuracy"] * self.result["num_losses"] + result["accuracy"]
        ) / (self.result["num_losses"] + 1)
        self.result["num_losses"] += 1

        # Store the results
        if prefix is None:
            result.pop("loss")
            result.pop("accuracy")
            self.result.update(result)
        else:
            self.result.update({f"{prefix}_{k}": v for k, v in result.items()})


def process_chain_config(self, dump_config=False, **parameters):
    """Process the full chain configuration and dump it.

    Parameters
    ----------
    dump_config : bool, default False
        Whether to dump the chain configuration in the log file or not
    **parameters : dict
        Dictionary of chain configuration parameters
    """
    # Store the modes for each step of the reconstruction. Make sure that
    # that the configuration is recognized.
    for module, valid_modes in self.modes.items():
        valid_modes = (None, "label", *valid_modes)
        assert module in parameters, (
            f"Must configure the {module} stage in the `chain` block. "
            f"The {module} mode should be one of {valid_modes}."
        )
        assert parameters[module] in valid_modes, (
            f"The {module} mode should be one of {valid_modes}. "
            f"Received '{parameters[module]}' instead."
        )
        setattr(self, module, parameters[module])

    # Do some logic checks on the chain parameters
    assert (
        not self.charge_rescaling or self.deghosting
    ), "Charge rescaling cannot be done without deghosting."
    assert self.charge_rescaling == self.deghosting or (
        self.deghosting == "uresnet" and self.charge_rescaling != "label"
    ), (
        "Label-based charge rescaling can only be done in conjunction "
        "with label-based deghosting."
    )
    assert not self.segmentation == "label" or self.deghosting in [None, "label"], (
        "Can only use labels for segmentation if labels are also used for "
        "remove ghost points."
    )
    assert (
        not self.point_proposal == "ppn" or self.segmentation == "uresnet"
    ), "For PPN to work, need the UResNet segmentation backbone."
    assert (
        not self.shower_primary == "gnn" or self.shower_aggregation == "gnn"
    ), "To use GNN for shower primaries, must aggregate showers with it."
    assert not self.particle_aggregation or (
        not self.shower_aggregation and not self.track_aggregation
    ), "Use particle aggregator or shower/track aggregators, not both."

    # Dump the chain configuration, if requested
    if dump_config:
        logger.info(f"Full chain configuration:")
        for k, v in parameters.items():
            v = v if v is not None else "null"
            logger.info(f"  {k:<27}: {v}")
        logger.info("")
