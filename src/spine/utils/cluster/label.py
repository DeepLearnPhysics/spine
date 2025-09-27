"""Class which adapts clustering labels given upstream semantic predictions."""

import numpy as np
import torch

from spine.data import TensorBatch
from spine.math.distance import METRICS, cdist, get_metric_id
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
from spine.utils.gnn.cluster import break_clusters
from spine.utils.torch.scripts import cdist_fast

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

    def __call__(self, clust_label, seg_label, seg_pred, ghost_pred=None):
        """Adapts the cluster labels for one entry or a batch of entries.

        Parameters
        ----------
        clust_label : Union[TensorBatch, np.ndarray, torch.Tensor]
            (N, N_l) Cluster label tensor
        seg_label : Union[TensorBatch, np.ndarray, torch.Tensor]
            (M, 5) Segmentation label tensor
        seg_pred : Union[TensorBatch, np.ndarray, torch.Tensor]
            (M/N_deghost) Segmentation predictions for each voxel
        ghost_pred : Union[TensorBatch, np.ndarray, torch.Tensor], optional
            (M) Ghost predictions for each voxel

        Returns
        -------
        Union[TensorBatch, np.ndarray, torch.Tensor]
            (N_deghost, N_l) Adapted cluster label tensor
        """
        # Set the data type/device based on the input
        ref_tensor = clust_label
        if isinstance(ref_tensor, TensorBatch):
            ref_tensor = ref_tensor.tensor
        self.torch = isinstance(ref_tensor, torch.Tensor)

        self.dtype = clust_label.dtype
        if self.torch:
            self.device = clust_label.device

        # Dispatch depending on the data type
        self._offset = 0
        if isinstance(clust_label, TensorBatch):
            # If it is batch data, call the main process function of each entry
            shape = (seg_pred.shape[0], clust_label.shape[1])
            clust_label_adapted = torch.empty(
                shape, dtype=clust_label.dtype, device=clust_label.device
            )
            for b in range(clust_label.batch_size):
                lower, upper = seg_pred.edges[b], seg_pred.edges[b + 1]
                ghost_pred_b = ghost_pred[b] if ghost_pred is not None else None
                clust_label_adapted[lower:upper] = self._process(
                    clust_label[b], seg_label[b], seg_pred[b], ghost_pred_b
                )

            return TensorBatch(clust_label_adapted, seg_pred.counts)

        else:
            # Otherwise, call the main process function directly
            return self._process(clust_label, seg_label, seg_pred, ghost_pred)

    def _process(self, clust_label, seg_label, seg_pred, ghost_pred=None):
        """Adapts the cluster labels for one entry or a batch of entries.

        Parameters
        ----------
        clust_label : Union[np.ndarray, torch.Tensor]
            (N, N_l) Cluster label tensor
        seg_label : Union[np.ndarray, torch.Tensor]
            (M, 5) Segmentation label tensor
        seg_pred : Union[np.ndarray, torch.Tensor]
            (M/N_deghost) Segmentation predictions for each voxel
        ghost_pred : Union[np.ndarray, torch.Tensor], optional
            (M) Ghost predictions for each voxel

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N_deghost, N_l) Adapted cluster label tensor
        """
        # If there are no points in this event, nothing to do
        coords = seg_label[:, :VALUE_COL]
        num_cols = clust_label.shape[1]
        if not len(coords):
            return self._ones((0, num_cols))

        # If there are no points after deghosting, nothing to do
        if ghost_pred is not None:
            deghost_index = self._where(ghost_pred == 0)[0]
            if not len(deghost_index):
                return self._ones((0, num_cols))

        # If there are no label points in this event, return dummy labels
        if not len(clust_label):
            if ghost_pred is None:
                shape = (len(coords), num_cols)
                dummy_labels = -self._ones(shape)
                dummy_labels[:, :VALUE_COL] = coords

            else:
                shape = (len(deghost_index), num_cols)
                dummy_labels = -self._ones(shape)
                dummy_labels[:, :VALUE_COL] = coords[deghost_index]

            return dummy_labels

        # Build a tensor of predicted segmentation that includes ghost points
        seg_label = self._to_long(seg_label[:, SHAPE_COL])
        if ghost_pred is not None and (len(ghost_pred) != len(seg_pred)):
            seg_pred_long = self._to_long(GHOST_SHP * self._ones(len(coords)))
            seg_pred_long[deghost_index] = seg_pred
            seg_pred = seg_pred_long

        # Prepare new labels
        new_label = -self._ones((len(coords), num_cols))
        new_label[:, :VALUE_COL] = coords

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

        true_deghost = seg_label < GHOST_SHP
        seg_mismatch = ~compat_mat[(seg_pred, seg_label)]
        new_label[true_deghost] = clust_label
        new_label[true_deghost & seg_mismatch, VALUE_COL:] = -self._ones(1)

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
            seg_clust_mask = compat_mat[s][self._to_long(clust_label[:, SHAPE_COL])]
            X_true = clust_label[seg_clust_mask]
            if len(X_true) == 0:
                continue

            # Loop over the set of unlabeled predicted points
            X_pred = coords[bad_index]
            tagged_voxels_count = 1
            while tagged_voxels_count > 0 and len(X_pred) > 0:
                # Compute Chebyshev distance between predicted and closest true
                distances = self._compute_distances(X_pred, X_true)
                distances, closest_ids = self._min(distances, 1)

                # Label unlabeled voxels that touch a compatible true voxel
                select_mask = distances < 1.1
                select_index = self._where(select_mask)[0]
                tagged_voxels_count = len(select_index)
                if tagged_voxels_count > 0:
                    # Use the label of the touching true voxel
                    additional_clust_label = self._cat(
                        [
                            X_pred[select_index],
                            X_true[closest_ids[select_index], VALUE_COL:],
                        ],
                        1,
                    )
                    new_label[bad_index[select_index]] = additional_clust_label

                    # Update the mask to not include the new assigned points
                    leftover_index = self._where(~select_mask)[0]
                    bad_index = bad_index[leftover_index]

                    # The new true available points are the ones we just added.
                    # The new pred points are those not yet labeled
                    X_true = additional_clust_label
                    X_pred = X_pred[leftover_index]

        # Remove predicted ghost points from the labels, set the shape
        # column of the label to the segmentation predictions.
        if ghost_pred is not None:
            new_label = new_label[deghost_index]
            new_label[:, SHAPE_COL] = seg_pred[deghost_index]
        else:
            new_label[:, SHAPE_COL] = seg_pred

        # Build a list of cluster indexes to break
        new_label_np = new_label
        if torch.is_tensor(new_label):
            new_label_np = new_label.detach().cpu().numpy()

        clusts = []
        labels = new_label_np[:, CLUST_COL]
        shapes = new_label_np[:, SHAPE_COL]
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
        new_label[:, CLUST_COL] = self._offset + break_clusters(
            new_label, clusts, self.break_eps, self.break_metric_id, self.break_p
        )
        self._offset = new_label[:, CLUST_COL].max() + 1

        return new_label

    def _where(self, x):
        if self.torch:
            return torch.where(x)
        else:
            return np.where(x)

    def _cat(self, x, axis):
        if self.torch:
            return torch.cat(x, axis)
        else:
            return np.concatenate(x, axis)

    def _ones(self, x):
        if self.torch:
            return torch.ones(x, dtype=self.dtype, device=self.device)
        else:
            return np.ones(x)

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
            return x.astype(int64)

    def _compute_distances(self, x, y):
        if self.torch:
            return cdist_fast(x[:, COORD_COLS], y[:, COORD_COLS], metric="chebyshev")
        else:
            return cdist(
                x[:, COORD_COLS], y[:, COORD_COLS], metric_id=METRICS["chebyshev"]
            )
