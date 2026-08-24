"""Class which adapts clustering labels given upstream semantic predictions."""

from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np

from spine.cluster.formation import break_clusters
from spine.constants import (
    DELTA_SHP,
    GHOST_SHP,
    MICHL_SHP,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.data import (
    ArrayLike,
    ClusterLabelBatch,
    ClusterLabelData,
    IndexBatch,
    TensorBatch,
    TensorData,
    TensorSchema,
)
from spine.math.distance import METRICS, get_metric_id
from spine.math.graph import bipartite_radius_graph, radius_graph, shortest_path
from spine.utils.conditional import TORCH_AVAILABLE, torch

__all__ = ["ClusterLabelAdapter"]


class ClusterLabelAdapter:
    """Adapts the cluster labels to account for the predicted semantics.

    Points wrongly predicted get the cluster label of the closest touching
    compatible cluster, if there is one. Points that are predicted as ghosts
    get invalid (-1) cluster labels everywhere.

    Equal-distance associations to distinct compatible instances remain
    invalid. Their ambiguous wavefront still propagates through the component,
    preventing the invalid target from acting as a geometric barrier.

    Instances that have been broken up by the deghosting or semantic
    segmentation process get assigned distinct cluster labels for each
    effective fragment, provided they appear in the `break_classes` list.

    Notes
    -----
    This class supports both Numpy arrays and Torch tensors.
    """

    class _AdaptationState(NamedTuple):
        """Event products shared while adapting individual semantic shapes."""

        clust_label: ClusterLabelData
        coords: ArrayLike
        seg_pred: ArrayLike
        new_features: ArrayLike
        seg_truth: ArrayLike
        true_deghost: ArrayLike
        compatible: ArrayLike
        mismatch: ArrayLike
        cluster_col: int

    def __init__(
        self,
        break_eps: float = 1.1,
        break_metric: str = "chebyshev",
        break_p: float = 2.0,
        break_classes: Sequence[int] = (
            SHOWR_SHP,
            TRACK_SHP,
            MICHL_SHP,
            DELTA_SHP,
        ),
        weighted: bool = True,
    ) -> None:
        """Initialize the adapter class.

        Parameters
        ----------
        break_eps : float, default 1.1
            Distance scale used in the breakup procedure.
        break_metric : str, default 'chebyshev'
            Distance metric used in the breakup procedure.
        break_p : float, default 2.
            p-norm factor for the Minkowski metric, if used.
        break_classes : Sequence[int], default shower, track, Michel and delta
            Semantic classes whose disconnected instances are split with
            DBSCAN.
        weighted : bool, default True
            Weight face, edge and corner propagation steps by their Euclidean
            lengths. If disabled, all 26 neighboring voxels cost one step.
        """
        # Store the connected-component configuration as one coherent unit.
        self.break_params = (
            break_eps,
            get_metric_id(break_metric, break_p),
            break_p,
        )
        self.break_classes = tuple(break_classes)
        self.weighted = weighted

        # Backend attributes are resolved from the batch at call time.
        self.torch = False
        self.dtype = np.dtype(np.float32)
        self.device = None
        self._offset = 0

    def __call__(
        self,
        clust_label: ClusterLabelBatch,
        seg_label: TensorBatch,
        seg_pred: TensorBatch,
        orig_index: IndexBatch | None = None,
    ) -> ClusterLabelBatch:
        """Adapt cluster labels through the callable interface.

        This delegates to :meth:`adapt`, which provides the explicitly named
        public operation for introspection and direct use.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Structured cluster labels for the input batch.
        seg_label : TensorBatch
            Semantic labels in the original voxel ordering.
        seg_pred : TensorBatch
            Semantic predictions before or after predicted deghosting.
        orig_index : IndexBatch, optional
            Original indexes retained by predicted deghosting.

        Returns
        -------
        ClusterLabelBatch
            Adapted structured cluster labels.
        """
        return self.adapt(clust_label, seg_label, seg_pred, orig_index)

    def adapt(
        self,
        clust_label: ClusterLabelBatch,
        seg_label: TensorBatch,
        seg_pred: TensorBatch,
        orig_index: IndexBatch | None = None,
    ) -> ClusterLabelBatch:
        """Adapt cluster labels for one batch.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Structured cluster labels for the input batch
        seg_label : TensorBatch
            (M, 5) Segmentation label tensor
        seg_pred : TensorBatch
            (M/N_deghost) Segmentation predictions for each voxel
        orig_index : IndexBatch, optional
            (N_deghost) Index of the deghosted voxels in the original input
            voxel ordering. This is used to map current predictions back into
            the original label space.

        Returns
        -------
        ClusterLabelBatch
            Adapted structured cluster labels
        """
        # Set the data type/device based on the compact association tensor.
        ref_tensor = clust_label.tensor
        self.torch = TORCH_AVAILABLE and isinstance(ref_tensor, torch.Tensor)
        self.dtype = clust_label.dtype
        if self.torch:
            self.device = clust_label.device

        # Adapt each event while retaining its compact particle table.
        self._offset = 0
        adapted = []
        for batch_id in range(clust_label.batch_size):
            orig_index_b = orig_index[batch_id] if orig_index is not None else None
            adapted.append(
                self._process(
                    clust_label[batch_id],
                    seg_label.event(batch_id),
                    seg_pred[batch_id],
                    orig_index_b,
                )
            )

        # Reassemble the adapted events into a single batch with the original particle tables.
        data = TensorBatch.from_data_list(adapted)
        return ClusterLabelBatch(data, clust_label.particles, clust_label.meta)

    def _process(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        seg_pred: ArrayLike,
        orig_index: ArrayLike | None = None,
    ) -> TensorData:
        """Adapt the cluster labels for one event.

        Parameters
        ----------
        clust_label : ClusterLabelData
            Structured cluster labels for one event.
        seg_label : TensorData
            Semantic labels for one event.
        seg_pred : Union[np.ndarray, torch.Tensor]
            (M/N_deghost) Segmentation predictions for each voxel
        orig_index : Union[np.ndarray, torch.Tensor], optional
            (N_deghost) Index of the deghosted voxels in the original input
            voxel ordering.

        Returns
        -------
        TensorData
            Adapted compact voxel labels for the event.
        """
        # Establish the output schema and resolve trivial events up front.
        coords = self._coordinates(seg_label)
        schema = ClusterLabelData.tensor_schema(clust_label.particles is not None)
        empty_result = self._empty_result(clust_label, seg_label, orig_index, schema)
        if empty_result is not None:
            return empty_result

        # Restore predictions to the original voxel domain before adapting truth.
        seg_pred = self._expand_prediction(seg_pred, coords, orig_index)
        true_deghost = self._to_long(seg_label.values) < GHOST_SHP
        self._validate_alignment(clust_label, seg_label, true_deghost)

        # Seed compatible truth rows, then repair semantic disagreements.
        new_features = self._initialize_features(clust_label, seg_label, seg_pred)
        self._adapt_mismatches(clust_label, seg_label, seg_pred, new_features)

        # Deghost only after adaptation, then split disconnected instances.
        coords, new_features, shapes = self._select_output(
            coords, new_features, seg_pred, orig_index
        )
        cluster_col = schema.feature_fields["cluster"][0]
        self._break_instances(coords, new_features, shapes, cluster_col)

        return TensorData(new_features, coords=coords, schema=schema)

    def _empty_result(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        orig_index: ArrayLike | None,
        schema: TensorSchema,
    ) -> TensorData | None:
        """Build an early result for an event without usable voxels.

        Parameters
        ----------
        clust_label : ClusterLabelData
            Structured cluster labels for one event.
        seg_label : TensorData
            Semantic labels for one event.
        orig_index : np.ndarray or torch.Tensor, optional
            Index of retained voxels after deghosting.
        schema : TensorSchema
            Schema of the adapted cluster-label product.

        Returns
        -------
        TensorData, optional
            Empty or dummy result when adaptation can stop immediately.
        """
        # Identify the number of features to preserve the output shape even for empty events.
        coords = self._coordinates(seg_label)
        num_features = clust_label.features.shape[1]

        # Preserve the feature width even for a genuinely empty input event.
        if len(coords) == 0:
            return TensorData(
                self._ones((0, num_features)),
                coords=coords,
                schema=schema,
            )

        # A fully deghosted event has no output rows to adapt.
        if orig_index is not None and len(orig_index) == 0:
            return TensorData(
                self._ones((0, num_features)),
                coords=coords[:0],
                schema=schema,
            )

        if len(clust_label) == 0:
            # Retained voxels still need correctly shaped invalid targets.
            output_coords = coords if orig_index is None else coords[orig_index]
            dummy_features = -self._ones((len(output_coords), num_features))
            return TensorData(dummy_features, coords=output_coords, schema=schema)

        return None

    def _expand_prediction(
        self,
        seg_pred: ArrayLike,
        coords: ArrayLike,
        orig_index: ArrayLike | None,
    ) -> ArrayLike:
        """Restore a deghosted prediction to the original voxel domain.

        Parameters
        ----------
        seg_pred : np.ndarray or torch.Tensor
            Semantic predictions, potentially restricted to retained voxels.
        coords : np.ndarray or torch.Tensor
            Coordinates in the original voxel ordering.
        orig_index : np.ndarray or torch.Tensor, optional
            Original indexes of the retained prediction rows.

        Returns
        -------
        np.ndarray or torch.Tensor
            Semantic predictions aligned with ``coords``.
        """
        # If the prediction is already in the original voxel domain, no expansion is needed.
        if orig_index is None or len(seg_pred) == len(coords):
            return seg_pred

        # Removed rows are known ghost predictions in the restored domain.
        expanded = self._to_long(GHOST_SHP * self._ones(len(coords)))
        expanded[orig_index] = seg_pred
        return expanded

    def _validate_alignment(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        true_deghost: ArrayLike,
    ) -> None:
        """Validate semantic-to-cluster truth row alignment.

        Parameters
        ----------
        clust_label : ClusterLabelData
            Compact labels for true non-ghost voxels.
        seg_label : TensorData
            Semantic labels in the original voxel ordering.
        true_deghost : np.ndarray or torch.Tensor
            Mask selecting true non-ghost semantic rows.

        Raises
        ------
        ValueError
            If the products differ in row count, coordinates or ordering.
        """
        # The compact cluster labels must contain exactly the true non-ghost voxels.
        if int(self._sum(true_deghost)) != len(clust_label):
            raise ValueError(
                "Cluster labels must contain exactly the true non-ghost voxels "
                "from the segmentation labels."
            )

        # Matching lengths are insufficient because propagation assumes the
        # same voxel ordering when it copies compact cluster features.
        seg_coords = self._coordinates(seg_label)
        clust_coords = self._coordinates(clust_label)
        if not self._equal(seg_coords[true_deghost], clust_coords):
            raise ValueError(
                "Cluster-label coordinates must match the ordered true "
                "non-ghost semantic coordinates."
            )

    def _initialize_features(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        seg_pred: ArrayLike,
    ) -> ArrayLike:
        """Initialize adapted features and retain compatible truth rows.

        Parameters
        ----------
        clust_label : ClusterLabelData
            Compact labels for true non-ghost voxels.
        seg_label : TensorData
            Semantic labels in the original voxel ordering.
        seg_pred : np.ndarray or torch.Tensor
            Semantic predictions aligned with ``seg_label``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Initially adapted feature matrix in the original voxel ordering.
        """
        # Identify the true non-ghost rows and their semantic disagreements.
        seg_truth = self._to_long(seg_label.values)
        true_deghost = seg_truth < GHOST_SHP
        compatible = self._compatibility_matrix()
        mismatch = ~compatible[(seg_pred, seg_truth)]

        # Start from an invalid canvas so ghosts and unresolved rows stay -1.
        new_features = -self._ones(
            (len(self._coordinates(seg_label)), clust_label.features.shape[1])
        )

        # Copy aligned truth features, then invalidate semantic disagreements.
        new_features[true_deghost] = clust_label.features
        new_features[true_deghost & mismatch] = -self._ones(1)
        return new_features

    def _compatibility_matrix(self) -> ArrayLike:
        """Build the allowed truth-to-prediction shape associations.

        Returns
        -------
        np.ndarray or torch.Tensor
            Boolean matrix indexed by predicted and true semantic shape.
        """
        # Tracks remain isolated, while the electromagnetic subclasses may
        # exchange labels with the parent shower class in either direction.
        compatible = self._eye(GHOST_SHP + 1)
        compatible[
            (
                [SHOWR_SHP, SHOWR_SHP, MICHL_SHP, DELTA_SHP],
                [MICHL_SHP, DELTA_SHP, SHOWR_SHP, SHOWR_SHP],
            )
        ] = True
        return compatible

    def _adapt_mismatches(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        seg_pred: ArrayLike,
        new_features: ArrayLike,
    ) -> None:
        """Assign compatible instance targets to semantic false positives.

        Parameters
        ----------
        clust_label : ClusterLabelData
            Compact labels for true non-ghost voxels.
        seg_label : TensorData
            Semantic labels in the original voxel ordering.
        seg_pred : np.ndarray or torch.Tensor
            Semantic predictions aligned with ``seg_label``.
        new_features : np.ndarray or torch.Tensor
            Adapted feature matrix to update in place.
        """
        # Identify the true non-ghost rows and their semantic disagreements.
        coords = self._coordinates(seg_label)
        seg_truth = self._to_long(seg_label.values)
        true_deghost = seg_truth < GHOST_SHP
        compatible = self._compatibility_matrix()
        mismatch = ~compatible[(seg_pred, seg_truth)]
        cluster_col = ClusterLabelData.tensor_schema(
            clust_label.particles is not None
        ).feature_fields["cluster"][0]

        # Bundle immutable event context shared by each predicted shape pass.
        state = self._AdaptationState(
            clust_label=clust_label,
            coords=coords,
            seg_pred=seg_pred,
            new_features=new_features,
            seg_truth=seg_truth,
            true_deghost=true_deghost,
            compatible=compatible,
            mismatch=mismatch,
            cluster_col=cluster_col,
        )

        # Each predicted class borrows labels only from compatible truth rows.
        for shape in self._unique(seg_pred):
            if shape == GHOST_SHP:
                continue
            self._adapt_shape(state, shape)

    def _adapt_shape(
        self,
        state: _AdaptationState,
        shape: int,
    ) -> None:
        """Adapt incompatible rows for one predicted semantic shape.

        Parameters
        ----------
        state : _AdaptationState
            Event products and masks shared across semantic shapes.
        shape : int
            Predicted semantic shape to process.
        """
        # Identify the predicted rows that disagree with the truth and need repair.
        bad_mask = (state.seg_pred == shape) & (~state.true_deghost | state.mismatch)
        bad_index = self._where(bad_mask)[0]
        if len(bad_index) == 0:
            return

        # Candidate sources must carry truth semantics compatible with shape.
        source_mask = state.compatible[shape][state.seg_truth[state.true_deghost]]
        source_coords = self._coordinates(state.clust_label)[source_mask]
        source_features = state.clust_label.features[source_mask]
        if len(source_coords) == 0:
            return

        # Tied geodesic fronts deliberately retain an invalid target.
        query_index, source_index = self._match_sources(
            state.coords[bad_index],
            source_coords,
            source_features[:, state.cluster_col],
        )
        state.new_features[bad_index[query_index]] = source_features[source_index]

    def _match_sources(
        self,
        query_coords: ArrayLike,
        source_coords: ArrayLike,
        source_labels: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Match queries to unambiguous compatible sources on the CPU.

        Parameters
        ----------
        query_coords : np.ndarray or torch.Tensor
            Coordinates requiring an adapted instance target.
        source_coords : np.ndarray or torch.Tensor
            Coordinates carrying compatible truth targets.
        source_labels : np.ndarray or torch.Tensor
            Instance identifier of each source coordinate.

        Returns
        -------
        np.ndarray or torch.Tensor
            Query rows that acquired an unambiguous target.
        np.ndarray or torch.Tensor
            Corresponding rows in the source feature matrix.
        """
        # Cross the CPU boundary once for each compact propagation problem.
        source_index, _, ambiguous = self._propagate(
            self._to_numpy(query_coords),
            self._to_numpy(source_coords),
            self._to_numpy(source_labels).astype(np.int64),
            weighted=self.weighted,
        )

        # Unreachable and tied rows deliberately retain invalid targets.
        query_index = np.where((source_index >= 0) & ~ambiguous)[0]
        source_index = source_index[query_index]

        # Convert only the compact association indexes back to the active
        # backend; coordinates and features remain in their original storage.
        if self.torch:
            query_index = torch.as_tensor(query_index, device=self.device)
            source_index = torch.as_tensor(source_index, device=self.device)

        return query_index, source_index

    @staticmethod
    def _propagate(
        query_coords: np.ndarray,
        source_coords: np.ndarray,
        source_labels: np.ndarray,
        weighted: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate source ownership using the configured edge costs.

        Parameters
        ----------
        query_coords : np.ndarray
            ``(N, 3)`` coordinates requiring a compatible target.
        source_coords : np.ndarray
            ``(M, 3)`` coordinates carrying compatible truth targets.
        source_labels : np.ndarray
            ``(M,)`` instance identifier associated with each source voxel.
        weighted : bool, default True
            Use Euclidean face, edge and corner step lengths. If disabled,
            assign unit cost to every 26-neighbor connection.

        Returns
        -------
        np.ndarray
            ``(N,)`` representative source-row index for each reached voxel.
        np.ndarray
            ``(N,)`` minimum graph distance from the source set.
        np.ndarray
            ``(N,)`` mask identifying equal-distance instance ambiguities.
        """
        # Propagation is only meaningful when both the query and source sets are non-empty.
        if len(query_coords) == 0 or len(source_coords) == 0:
            return (
                np.full(len(query_coords), -1, dtype=np.int64),
                np.full(len(query_coords), -1.0, dtype=np.float64),
                np.zeros(len(query_coords), dtype=np.bool_),
            )

        # Construct the 26-neighbor graph and identify touching truth sources.
        source_edges = bipartite_radius_graph(
            query_coords,
            source_coords,
            1.1,
            metric_id=METRICS["chebyshev"],
        )
        if len(source_edges) == 0:
            return (
                np.full(len(query_coords), -1, dtype=np.int64),
                np.full(len(query_coords), -1.0, dtype=np.float64),
                np.zeros(len(query_coords), dtype=np.bool_),
            )

        source_weights = np.ones(len(source_edges), dtype=np.float64)
        if weighted:
            source_weights = np.linalg.norm(
                query_coords[source_edges[:, 0]] - source_coords[source_edges[:, 1]],
                axis=1,
            )
        owners = np.unique(source_labels[source_edges[:, 1]])

        edge_index = radius_graph(
            query_coords,
            1.1,
            metric_id=METRICS["chebyshev"],
            use_hash=True,
        )
        edge_weights = np.ones(len(edge_index), dtype=np.float64)
        if weighted:
            edge_weights = np.linalg.norm(
                query_coords[edge_index[:, 0]] - query_coords[edge_index[:, 1]],
                axis=1,
            )

        # Compute paths independently so exact ties between owners survive.
        return ClusterLabelAdapter._resolve_owner_paths(
            *ClusterLabelAdapter._owner_paths(
                (source_edges, source_weights),
                (edge_index, edge_weights, len(query_coords)),
                source_labels,
                owners,
            )
        )

    @staticmethod
    def _owner_paths(
        source_graph: tuple[np.ndarray, np.ndarray],
        query_graph: tuple[np.ndarray, np.ndarray, int],
        source_labels: np.ndarray,
        owners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute weighted shortest paths independently for each owner.

        Parameters
        ----------
        source_graph : tuple[np.ndarray, np.ndarray]
            Direct query-to-source edges and their initial costs.
        query_graph : tuple[np.ndarray, np.ndarray, int]
            Query edges, edge weights and total node count.
        source_labels : np.ndarray
            ``(M,)`` instance identifier associated with each source voxel.
        owners : np.ndarray
            Source owners that touch at least one query voxel.

        Returns
        -------
        np.ndarray
            ``(O, N)`` shortest query distances for each owner.
        np.ndarray
            ``(O, N)`` representative source-row indexes for each path.
        """
        source_edges, source_weights = source_graph
        edge_index, edge_weights, num_nodes = query_graph
        owner_distances = []
        owner_sources = []
        for owner in owners:
            owner_mask = source_labels[source_edges[:, 1]] == owner
            distances, sources = shortest_path(
                edge_index,
                edge_weights,
                num_nodes,
                (
                    source_edges[owner_mask, 0],
                    source_weights[owner_mask],
                    source_edges[owner_mask, 1],
                ),
                directed=False,
            )
            owner_distances.append(distances)
            owner_sources.append(sources)

        return np.stack(owner_distances), np.stack(owner_sources)

    @staticmethod
    def _resolve_owner_paths(
        distances: np.ndarray,
        sources: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select the minimum owner distance and identify weighted ties.

        Parameters
        ----------
        distances : np.ndarray
            ``(O, N)`` shortest query distances for each owner.
        sources : np.ndarray
            ``(O, N)`` representative source-row indexes for each owner.

        Returns
        -------
        np.ndarray
            ``(N,)`` representative source-row index for each reached voxel.
        np.ndarray
            ``(N,)`` minimum weighted graph distance from the source set.
        np.ndarray
            ``(N,)`` mask identifying equal-distance instance ambiguities.
        """
        distance = np.min(distances, axis=0)
        reached = np.isfinite(distance)
        closest_owner = np.argmin(distances, axis=0)
        source = sources[closest_owner, np.arange(distances.shape[1])]
        ambiguous = np.sum(np.isclose(distances, distance), axis=0) > 1

        source[~reached] = -1
        distance[~reached] = -1.0
        return source, distance, ambiguous & reached

    @staticmethod
    def _select_output(
        coords: ArrayLike,
        new_features: ArrayLike,
        seg_pred: ArrayLike,
        orig_index: ArrayLike | None,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Select deghosted output rows and their predicted shapes.

        Parameters
        ----------
        coords : np.ndarray or torch.Tensor
            Coordinates in the original voxel ordering.
        new_features : np.ndarray or torch.Tensor
            Adapted features in the original voxel ordering.
        seg_pred : np.ndarray or torch.Tensor
            Semantic predictions in the original voxel ordering.
        orig_index : np.ndarray or torch.Tensor, optional
            Original indexes retained by predicted deghosting.

        Returns
        -------
        tuple[np.ndarray or torch.Tensor, ...]
            Coordinates, features and shapes restricted to retained voxels.
        """
        if orig_index is None:
            return coords, new_features, seg_pred
        return coords[orig_index], new_features[orig_index], seg_pred[orig_index]

    def _break_instances(
        self,
        coords: ArrayLike,
        new_features: ArrayLike,
        shapes: ArrayLike,
        cluster_col: int,
    ) -> None:
        """Assign distinct IDs to disconnected pieces of each instance.

        Parameters
        ----------
        coords : np.ndarray or torch.Tensor
            Coordinates retained after predicted deghosting.
        new_features : np.ndarray or torch.Tensor
            Adapted features to update in place.
        shapes : np.ndarray or torch.Tensor
            Predicted semantic shape of every retained voxel.
        cluster_col : int
            Feature column containing the instance identifier.
        """
        # Candidate discovery is inexpensive on NumPy and shared by backends.
        features_numpy = self._to_numpy(new_features)
        shapes_numpy = self._to_numpy(shapes)
        labels = features_numpy[:, cluster_col]
        clusters = self._break_candidates(labels, shapes_numpy)

        # The decorated cluster utility preserves the input storage backend.
        broken_labels = break_clusters(
            coords,
            new_features[:, cluster_col],
            clusters,
            *self.break_params,
        )

        # Keep invalid labels at -1 while making valid IDs unique across events.
        broken_valid = broken_labels >= 0
        broken_labels[broken_valid] += self._offset
        new_features[:, cluster_col] = broken_labels

        valid = new_features[:, cluster_col] >= 0
        if self._sum(valid):
            self._offset = int(new_features[valid, cluster_col].max()) + 1

    def _break_candidates(
        self,
        labels: np.ndarray,
        shapes: np.ndarray,
    ) -> list[np.ndarray]:
        """Build cluster indexes subject to connected-component tests.

        Parameters
        ----------
        labels : np.ndarray
            Adapted cluster identifier of every retained voxel.
        shapes : np.ndarray
            Predicted semantic shape of every retained voxel.

        Returns
        -------
        list[np.ndarray]
            Voxel-index arrays to split into connected components.
        """
        clusters = []

        # Build connected-component candidates separately for every semantic
        # class so that an instance ID reused across classes cannot mix them.
        for break_class in self.break_classes:
            class_index = np.where(shapes == break_class)[0]
            class_labels = labels[class_index]
            for cluster_id in np.unique(class_labels):
                if cluster_id < 0:
                    continue
                clusters.append(class_index[class_labels == cluster_id])

        return clusters

    @staticmethod
    def _coordinates(data: TensorData | ClusterLabelData) -> ArrayLike:
        """Return the required coordinate matrix of a voxel product.

        Parameters
        ----------
        data : TensorData or ClusterLabelData
            Coordinate-bearing event product.

        Returns
        -------
        np.ndarray or torch.Tensor
            Coordinate matrix associated with the product.

        Raises
        ------
        ValueError
            If the product does not carry coordinates.
        """
        coords = data.coords
        if coords is None:
            raise ValueError("Cluster-label adaptation requires voxel coordinates.")
        return coords

    def _to_numpy(self, array: ArrayLike) -> np.ndarray:
        """Return a validated NumPy representation of a backend-native array.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Array to expose to CPU-only adaptation routines. Its backend must
            agree with the backend selected for the current batch.

        Returns
        -------
        np.ndarray
            CPU representation of the input array. NumPy inputs are returned
            without an unnecessary cast or copy.

        Raises
        ------
        TypeError
            If the array backend disagrees with the current batch backend.
        """
        if self.torch:
            if not isinstance(array, torch.Tensor):
                raise TypeError("Expected a Torch tensor for this adapter batch.")
            tensor = cast(torch.Tensor, array)
            return tensor.detach().cpu().numpy()

        if not isinstance(array, np.ndarray):
            raise TypeError("Expected a NumPy array for this adapter batch.")
        return array

    def _where(self, array: ArrayLike) -> tuple[ArrayLike, ...]:
        """Return indexes where a backend-native condition is true."""
        if self.torch:
            return torch.where(array)
        return np.where(array)

    def _ones(self, shape: int | tuple[int, ...]) -> ArrayLike:
        """Create backend-native ones with the adapter feature dtype."""
        if self.torch:
            return torch.ones(shape, dtype=self.dtype, device=self.device)
        return np.ones(shape, dtype=self.dtype)

    def _eye(self, size: int) -> ArrayLike:
        """Create a backend-native boolean identity matrix."""
        if self.torch:
            return torch.eye(size, dtype=torch.bool, device=self.device)
        return np.eye(size, dtype=bool)

    def _unique(self, array: ArrayLike) -> ArrayLike:
        """Return sorted unique values as backend-native integers."""
        if self.torch:
            tensor = cast(torch.Tensor, array)
            return torch.unique(tensor).long()
        return np.unique(array).astype(np.int64)

    def _to_long(self, array: ArrayLike) -> ArrayLike:
        """Cast an array to its backend-native 64-bit integer dtype."""
        if self.torch:
            return cast(torch.Tensor, array).long()
        return array.astype(np.int64)

    def _sum(self, array: ArrayLike) -> int | float:
        """Reduce an array to a backend-independent scalar sum."""
        if self.torch:
            return array.sum().item()
        return np.sum(array)

    def _equal(self, first: ArrayLike, second: ArrayLike) -> bool:
        """Test two backend-native arrays for exact equality."""
        if self.torch:
            return torch.equal(first, second)
        return np.array_equal(first, second)
