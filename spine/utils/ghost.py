"""Algorithms associated with the deghosting process."""

import numpy as np
import torch
from torch_cluster import knn
from scipy.spatial.distance import cdist

from spine.data import TensorBatch
from spine.utils.numba_local import dbscan

from .globals import (
        COORD_COLS, VALUE_COL, CLUST_COL, SHAPE_COL, SHOWR_SHP, TRACK_SHP,
        MICHL_SHP, DELTA_SHP, LOWES_SHP, GHOST_SHP)


def compute_rescaled_charge_batch(data, collection_only=False, collection_id=2):
    """Batched version of :func:`compute_rescaled_charge`.

    Parameters
    ----------
    data : TensorBatch
        (N, 1 + D + N_f + 6) tensor of voxel/value pairs
    collection_only : bool, default False
        Only use the collection plane to estimate the rescaled charge
    collection_id : int, default 2
        Index of the collection plane

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (N) Rescaled charge values
    """
    charges = data._empty(len(data.tensor))
    for b in range(data.batch_size):
        lower, upper = data.edges[b], data.edges[b+1]
        charges[lower:upper] = compute_rescaled_charge(
                data[b], collection_only, collection_id)

    return charges


def adapt_labels_batch(clust_label, seg_label, seg_pred, ghost_pred=None,
                       break_classes=[SHOWR_SHP,TRACK_SHP,MICHL_SHP,DELTA_SHP],
                       break_eps=1.1, break_metric='chebyshev'):
    """Batched version of :func:`adapt_labels`.

    Parameters
    ----------
    clust_label : TensorBatch
        (N, N_l) Cluster label tensor
    seg_label : TensorBatch
        (M, 5) Segmentation label tensor
    seg_pred : TensorBatch
        (M/N_deghost) Segmentation predictions for each voxel
    ghost_pred : TensorBatch, optional
        (M) Ghost predictions for each voxel
    break_classes : List[int], default 
                    [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP]
        Classes to run DBSCAN on to break up
    break_eps : float, default 1.1
        Distance scale used in the break up procedure
    break_metric : str, default 'chebyshev'
        Distance metric used in the break up produce

    Returns
    -------
    TensorBatch
        (N_deghost, N_l) Adapted cluster label tensor
    """
    shape = (seg_pred.shape[0], clust_label.shape[1])
    clust_label_adapted = torch.empty(
            shape, dtype=clust_label.dtype, device=clust_label.device)
    for b in range(clust_label.batch_size):
        lower, upper = seg_pred.edges[b], seg_pred.edges[b+1]
        ghost_pred_b = ghost_pred[b] if ghost_pred is not None else None
        clust_label_adapted[lower:upper] = adapt_labels(
                clust_label[b], seg_label[b], seg_pred[b],
                ghost_pred_b, break_classes, break_eps, break_metric)

    return TensorBatch(clust_label_adapted, seg_pred.counts)


def compute_rescaled_charge(data, collection_only=False, collection_id=2):
    """Computes rescaled charge after deghosting.

    The last 6 columns of the input tensor *MUST* contain:
    - charge in each of the projective planes (3)
    - index of the hit in each 2D projection (3)

    Notes
    -----
    This function should work on numpy arrays or Torch tensors.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        (N, 1 + D + N_f + 6) tensor of voxel/value pairs
    collection_only : bool, default False
        Only use the collection plane to estimate the rescaled charge
    collection_id : int, default 2
        Index of the collection plane

    Returns
    -------
    data : Union[np.ndarray, torch.Tensor]
        (N) Rescaled charge values
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(data):
        unique = torch.unique
        empty = lambda shape: torch.empty(shape, dtype=torch.long,
                device=data.device)
        sum = lambda x: torch.sum(x, dim=1)
    else:
        unique = np.unique
        empty = np.empty
        sum = lambda x: np.sum(x, axis=1)

    # Count how many times each wire hit is used to form a space point
    hit_ids = data[:, -3:]
    _, inverse, counts = unique(
            hit_ids, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse].reshape(-1, 3)

    # Rescale the charge on the basis of hit multiplicity
    hit_charges = data[:, -6:-3]
    if not collection_only:
        # Take the average of the charge estimates from each active plane
        pmask   = hit_ids > -1
        charges = sum((hit_charges*pmask)/multiplicity)/sum(pmask)
    else:
        # Only use the collection plane measurement
        charges = hit_charges[:, collection_id]/multiplicity[:, collection_id]

    return charges


def adapt_labels(clust_label, seg_label, seg_pred, ghost_pred=None,
                 break_classes=[SHOWR_SHP,TRACK_SHP,MICHL_SHP,DELTA_SHP],
                 break_eps=1.1, break_metric='chebyshev'):
    """Adapts the cluster labels to account for the predicted semantics.

    Points wrongly predicted get the cluster label of the closest touching
    cluster, if there is one. Points that are predicted as ghosts get invalid
    (-1) cluster labels everywhere.

    Instances that have been broken up by the deghosting process get assigned
    distinct cluster labels for each effective fragment.

    Notes
    -----
    This function should work on Numpy arrays or Torch tensors.

    Uses GPU version from `torch_cluster.knn` to speed up the label adaptation
    computation.

    Parameters
    ----------
    clust_label : Union[np.ndarray, torch.Tensor]
        (N, N_l) Cluster label tensor
    seg_label : List[Union[np.ndarray, torch.Tensor]]
        (M, 5) Segmentation label tensor
    seg_pred : Union[np.ndarray, torch.Tensor]
        (M/N_deghost) Segmentation predictions for each voxel
    ghost_pred : Union[np.ndarray, torch.Tensor], optional
        (M) Ghost predictions for each voxel
    break_classes : List[int], default 
                    [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP]
        Classes to run DBSCAN on to break up
    break_eps : float, default 1.1
        Distance scale used in the break up procedure
    break_metric : str, default 'chebyshev'
        Distance metric used in the break up produce

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (N_deghost, N_l) Adapted cluster label tensor
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(seg_label):
        dtype, device = clust_label.dtype, clust_label.device
        where, cat, argmax = torch.where, torch.cat, torch.amax
        ones    = lambda x: torch.ones(x, dtype=dtype, device=device)
        eye     = lambda x: torch.eye(x, dtype=torch.bool, device=device)
        unique  = lambda x: torch.unique(x).long()
        to_long = lambda x: x.long()
        to_bool = lambda x: x.bool()
        compute_neighbor = lambda x, y: knn(
                y[:, COORD_COLS], x[:, COORD_COLS], 1)[1]
        compute_distances = lambda x, y: torch.amax(
                torch.abs(y[:, COORD_COLS] - x[:, COORD_COLS]), dim=1)

    else:
        where, cat, argmax = np.where, np.concatenate, np.argmax
        ones    = lambda x: np.ones(x, dtype=clust_label.dtype)
        eye     = lambda x: np.eye(x, dtype=bool)
        unique  = lambda x: np.unique(x).astype(np.int64)
        to_long = lambda x: x.astype(np.int64)
        to_bool = lambda x: x.astype(bool)
        compute_neighbor = lambda x, y: cdist(
                x[:, COORD_COLS], y[:, COORD_COLS]).argmin(axis=1)
        compute_distances = lambda x, y: np.amax(
                np.abs(x[:, COORD_COLS] - y[:, COORD_COLS]), axis=1)

    # If there are no points in this event, nothing to do
    coords = seg_label[:, :VALUE_COL]
    num_cols = clust_label.shape[1]
    if not len(coords):
        return ones((0, num_cols))

    # If there are no points after deghosting, nothing to do
    if ghost_pred is not None:
        deghost_index = where(ghost_pred == 0)[0]
        if not len(deghost_index):
            return ones((0, num_cols))

    # If there are no label poins in this event, return dummy labels
    if not len(clust_label):
        if ghost_pred is None:
            shape = (len(coords), num_cols)
            dummy_labels = -1 * ones(shape)
            dummy_labels[:, :VALUE_COL] = coords

        else:
            shape = (len(deghost_index), num_cols)
            dummy_labels = -1 * ones(shape)
            dummy_labels[:, :VALUE_COL] = coords[deghost_index]

        return dummy_labels

    # Build a tensor of predicted segmentation that includes ghost points
    seg_label = to_long(seg_label[:, SHAPE_COL])
    if ghost_pred is not None and (len(ghost_pred) != len(seg_pred)):
        seg_pred_long = to_long(GHOST_SHP*ones(len(coords)))
        seg_pred_long[deghost_index] = seg_pred
        seg_pred = seg_pred_long

    # Prepare new labels
    new_label = -1. * ones((len(coords), num_cols))
    new_label[:, :VALUE_COL] = coords

    # Check if the segment labels and predictions are compatible. If they are
    # compatible, store the cluster labels as is. Track points do not mix
    # with other classes, but EM classes are allowed to.
    compat_mat = eye(GHOST_SHP + 1)
    compat_mat[([SHOWR_SHP, SHOWR_SHP, MICHL_SHP, DELTA_SHP],
                [MICHL_SHP, DELTA_SHP, SHOWR_SHP, SHOWR_SHP])] = True

    true_deghost = seg_label < GHOST_SHP
    seg_mismatch = ~compat_mat[(seg_pred, seg_label)]
    new_label[true_deghost] = clust_label
    new_label[true_deghost & seg_mismatch, VALUE_COL:] = -1.

    # For mismatched predictions, attempt to find a touching instance of the
    # same class to assign it sensible cluster labels.
    for s in unique(seg_pred):
        # Skip predicted ghosts (they keep their invalid labels)
        if s == GHOST_SHP:
            continue

        # Restrict to points in this class that have incompatible segment
        # labels. Track points do not mix, EM points are allowed to. 
        bad_index = where((seg_pred == s) & (~true_deghost | seg_mismatch))[0]
        if len(bad_index) == 0:
            continue

        # Find points in clust_label that have compatible segment labels
        if s == TRACK_SHP or s == LOWES_SHP:
            seg_clust_mask = clust_label[:, SHAPE_COL] == s
        else:
            seg_clust_mask = (
                    (clust_label[:, SHAPE_COL] != TRACK_SHP) &
                    (clust_label[:, SHAPE_COL] != LOWES_SHP))

        X_true = clust_label[seg_clust_mask]
        if len(X_true) == 0:
            continue

        # Loop over the set of unlabeled predicted points
        X_pred = coords[bad_index]
        tagged_voxels_count = 1
        while tagged_voxels_count > 0 and len(X_pred) > 0:
            # Find the nearest neighbor to each predicted point
            closest_ids = compute_neighbor(X_pred, X_true)

            # Compute Chebyshev distance between predicted and closest true.
            distances = compute_distances(X_pred, X_true[closest_ids])

            # Label unlabeled voxels that touch a compatible true voxel
            select_mask = distances <= 1
            select_index = where(select_mask)[0]
            tagged_voxels_count = len(select_index)
            if tagged_voxels_count > 0:
                # Use the label of the touching true voxel
                additional_clust_label = cat(
                        [X_pred[select_index], 
                         X_true[closest_ids[select_index], VALUE_COL:]], 1)
                new_label[bad_index[select_index]] = additional_clust_label

                # Update the mask to not include the new assigned points
                leftover_index = where(~select_mask)[0]
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

    # Now if an instance was broken up, assign it different cluster IDs
    cluster_count = int(clust_label[:, CLUST_COL].max()) + 1
    for break_class in break_classes:
        # Restrict to the set of labels associated with this class
        break_index = where(new_label[:, SHAPE_COL] == break_class)[0]
        restricted_label = new_label[break_index]
        restricted_coordinates = restricted_label[:, COORD_COLS]

        # Loop over true cluster instances in the new label tensor, break
        for c in unique(restricted_label[:, CLUST_COL]):
            # Skip invalid cluster ID
            if c < 0:
                continue

            # Restrict tensor to a specific cluster, get voxel coordinates
            cluster_index = where(restricted_label[:, CLUST_COL] == c)[0]
            coordinates = restricted_coordinates[cluster_index]
            if torch.is_tensor(coordinates):
                coordinates = coordinates.detach().cpu().numpy()

            # Run DBSCAN on the cluster, update labels
            break_labels = dbscan(
                    coordinates, eps=break_eps, metric=break_metric)
            break_labels += cluster_count
            if torch.is_tensor(new_label):
                break_labels = torch.tensor(break_labels,
                        dtype=new_label.dtype, device=new_label.device)
            new_label[break_index[cluster_index], CLUST_COL] = break_labels
            cluster_count = int(break_labels.max()) + 1

    return new_label
