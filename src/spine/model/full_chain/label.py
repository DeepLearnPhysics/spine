"""Class which adapts clustering labels given upstream semantic predictions."""

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
    ClusterLabelBatch,
    ClusterLabelData,
    TensorBatch,
    TensorData,
)
from spine.math.distance import METRICS, cdist, get_metric_id
from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.utils.torch.runtime import cdist_fast

__all__ = ["ClusterLabelAdapter"]


class ClusterLabelAdapter:
    """Adapts the cluster labels to account for the predicted semantics.

    Points wrongly predicted get the cluster label of the closest touching
    compatible cluster, if there is one. Points that are predicted as ghosts
    get invalid (-1) cluster labels everywhere.

    Instances that have been broken up by the deghosting or semantic
    segmentation process get assigned distinct cluster labels for each
    effective fragment, provided they appear in the `break_classes` list.

    Notes
    -----
    This class supports both Numpy arrays and Torch tensors.
    """

    def __init__(
        self,
        break_eps=1.1,
        break_metric="chebyshev",
        break_p=2.0,
        break_classes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
    ):
        """Initialize the adapter class.

        Parameters
        ----------
        dtype : str, default 'torch'
            Type of data to be processed through the label adapter
        break_eps : float, default 1.1
            Distance scale used in the break up procedure
        break_metric : str, default 'chebyshev'
            Distance metric used in the break up produce
        p : float, default 2.
            p-norm factor for the Minkowski metric, if used
        break_classes : List[int], default
                        [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP]
            Classes to run DBSCAN on to break up
        """
        # Store relevant parameters
        self.break_eps = break_eps
        self.break_metric_id = get_metric_id(break_metric, break_p)
        self.break_p = break_p
        self.break_classes = break_classes

        # Attributes used to fetch the correct functions
        self.torch, self.dtype, self.device = None, None, None

    def __call__(
        self,
        clust_label: ClusterLabelBatch,
        seg_label: TensorBatch,
        seg_pred: TensorBatch,
        orig_index=None,
    ) -> ClusterLabelBatch:
        """Adapts the cluster labels for one entry or a batch of entries.

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

        data = TensorBatch.from_data_list(adapted)
        return ClusterLabelBatch(data, clust_label.particles, clust_label.meta)

    def _process(
        self,
        clust_label: ClusterLabelData,
        seg_label: TensorData,
        seg_pred,
        orig_index=None,
    ) -> TensorData:
        """Adapts the cluster labels for one entry or a batch of entries.

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
        # Resolve the two products into unambiguous coordinate/feature arrays.
        coords = seg_label.coords
        seg_truth = self._to_long(seg_label.values)
        clust_coords = clust_label.coords
        clust_features = clust_label.features
        num_features = clust_features.shape[1]
        schema = ClusterLabelData.tensor_schema(clust_label.particles is not None)

        # If there are no points in this event, nothing to do.
        if len(coords) == 0:
            return TensorData(
                self._ones((0, num_features)), coords=coords, schema=schema
            )

        deghost_index = orig_index

        # If there are no points after deghosting, nothing to do
        if deghost_index is not None:
            if len(deghost_index) == 0:
                return TensorData(
                    self._ones((0, num_features)),
                    coords=coords[:0],
                    schema=schema,
                )

        # If there are no label points in this event, return dummy labels
        if len(clust_label) == 0:
            output_coords = coords if deghost_index is None else coords[deghost_index]
            dummy_features = -self._ones((len(output_coords), num_features))
            return TensorData(dummy_features, coords=output_coords, schema=schema)

        # Build a tensor of predicted segmentation that includes ghost points
        if deghost_index is not None and (len(seg_pred) != len(coords)):
            seg_pred_long = self._to_long(GHOST_SHP * self._ones(len(coords)))
            seg_pred_long[deghost_index] = seg_pred
            seg_pred = seg_pred_long

        # Prepare invalid feature rows, then retain compatible truth rows.
        new_features = -self._ones((len(coords), num_features))

        # Check if the segment labels and predictions are compatible. If they are
        # compatible, store the cluster labels as is. Track points do not mix
        # with other classes, but EM classes are allowed to.
        compat_mat = self._eye(GHOST_SHP + 1)
        compat_mat[
            (
                [SHOWR_SHP, SHOWR_SHP, MICHL_SHP, DELTA_SHP],
                [MICHL_SHP, DELTA_SHP, SHOWR_SHP, SHOWR_SHP],
            )
        ] = True

        true_deghost = seg_truth < GHOST_SHP
        if int(self._sum(true_deghost)) != len(clust_label):
            raise ValueError(
                "Cluster labels must contain exactly the true non-ghost voxels "
                "from the segmentation labels."
            )
        clust_shapes = seg_truth[true_deghost]
        seg_mismatch = ~compat_mat[(seg_pred, seg_truth)]
        new_features[true_deghost] = clust_features
        new_features[true_deghost & seg_mismatch] = -self._ones(1)

        # For mismatched predictions, attempt to find a touching instance of the
        # same class to assign it sensible cluster labels.
        for s in self._unique(seg_pred):
            # Skip predicted ghosts (they keep their invalid labels)
            if s == GHOST_SHP:
                continue

            # Restrict to points in this class that have incompatible segment
            # labels. Track points do not mix, EM points are allowed to.
            bad_index = self._where((seg_pred == s) & (~true_deghost | seg_mismatch))[0]
            if len(bad_index) == 0:
                continue

            # Find points in clust_label that have compatible segment labels
            seg_clust_mask = compat_mat[s][clust_shapes]
            true_coords = clust_coords[seg_clust_mask]
            true_features = clust_features[seg_clust_mask]
            if len(true_coords) == 0:
                continue

            # Loop over the set of unlabeled predicted points
            X_pred = coords[bad_index]
            tagged_voxels_count = 1
            while tagged_voxels_count > 0 and len(X_pred) > 0:
                # Compute Chebyshev distance between predicted and closest true
                distances = self._compute_distances(X_pred, true_coords)
                distances, closest_ids = self._min(distances, 1)

                # Label unlabeled voxels that touch a compatible true voxel
                select_mask = distances < 1.1
                select_index = self._where(select_mask)[0]
                tagged_voxels_count = len(select_index)
                if tagged_voxels_count > 0:
                    # Use the label of the touching true voxel
                    additional_features = true_features[closest_ids[select_index]]
                    new_features[bad_index[select_index]] = additional_features

                    # Update the mask to not include the new assigned points
                    leftover_index = self._where(~select_mask)[0]
                    bad_index = bad_index[leftover_index]

                    # The new true available points are the ones we just added.
                    # The new pred points are those not yet labeled
                    true_coords = X_pred[select_index]
                    true_features = additional_features
                    X_pred = X_pred[leftover_index]

        # Remove predicted ghost points. Semantic predictions remain a
        # separate product and are never packed into the cluster labels.
        if deghost_index is not None:
            coords = coords[deghost_index]
            new_features = new_features[deghost_index]

        # Build a list of cluster indexes to break
        cluster_col = schema.feature_fields["cluster"][0]
        new_features_np = new_features
        if self.torch:
            new_features_np = new_features.detach().cpu().numpy()

        clusts = []
        labels = new_features_np[:, cluster_col]
        shapes = seg_pred[deghost_index] if deghost_index is not None else seg_pred
        if self.torch:
            shapes = shapes.detach().cpu().numpy()
        for break_class in self.break_classes:
            index_s = np.where(shapes == break_class)[0]
            labels_s = labels[index_s]
            for c in np.unique(labels_s):
                # If the cluster ID is invalid, skip
                if c < 0:
                    continue

                # Append cluster
                clusts.append(index_s[labels_s == c])

        # Now if an instance was broken up, assign it different cluster IDs
        broken_labels = break_clusters(
            coords,
            new_features[:, cluster_col],
            clusts,
            self.break_eps,
            self.break_metric_id,
            self.break_p,
        )
        broken_valid = broken_labels >= 0
        broken_labels[broken_valid] += self._offset
        new_features[:, cluster_col] = broken_labels
        valid = new_features[:, cluster_col] >= 0
        if self._sum(valid):
            self._offset = int(new_features[valid, cluster_col].max()) + 1

        return TensorData(new_features, coords=coords, schema=schema)

    def _where(self, x):
        if self.torch:
            return torch.where(x)
        else:
            return np.where(x)

    def _ones(self, x):
        if self.torch:
            return torch.ones(x, dtype=self.dtype, device=self.device)
        else:
            return np.ones(x, dtype=self.dtype)

    def _eye(self, x):
        if self.torch:
            return torch.eye(x, dtype=torch.bool, device=self.device)
        else:
            return np.eye(x, dtype=bool)

    def _min(self, x, axis):
        if self.torch:
            return torch.min(x, axis)
        else:
            return np.min(x, axis), np.argmin(x, axis)

    def _unique(self, x):
        if self.torch:
            return torch.unique(x).long()
        else:
            return np.unique(x).astype(np.int64)

    def _to_long(self, x):
        if self.torch:
            return x.long()
        else:
            return x.astype(np.int64)

    def _sum(self, x):
        if self.torch:
            return x.sum().item()
        return np.sum(x)

    def _compute_distances(self, x, y):
        if self.torch:
            return cdist_fast(x, y, metric="chebyshev")
        else:
            return cdist(x, y, metric_id=METRICS["chebyshev"])
