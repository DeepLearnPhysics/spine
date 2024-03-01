import numpy as np
import torch
from scipy.special import softmax
from scipy.spatial.distance import cdist

# TODO: remove utils.py so that we can do local imports
from mlreco.utils import numba_local as nbl
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.data_structures import TensorBatch
from mlreco.utils.globals import (
        BATCH_COL, COORD_COLS, PPN_ROFF_COLS, PPN_RTYPE_COLS, PPN_RPOS_COLS,
        PPN_END_COLS, TRACK_SHP, LOWES_SHP, UNKWN_SHP)


def get_ppn_labels(particle_v, meta, dim=3, min_voxel_count=5,
                   min_energy_deposit=0, include_point_tagging=True):
    """Gets particle point coordinates and informations for running PPN.

    We skip some particles under specific conditions (e.g. low energy deposit,
    low voxel count, nucleus track, etc.)

    Parameters
    ----------
    particle_v : List[larcv.Particle]
        List of LArCV particle objects in the image
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    dim : int, default 3
        Number of dimensions of the image
    min_voxel_count : int, default 5
        Minimum number of voxels associated with a particle to be included
    min_energy_deposit : float, default 0
        Minimum energy deposition associated with a particle to be included
    include_point_tagging : bool, default True
        If True, include an a label of 0 for start points and 1 for end points

    Returns
    -------
    np.array
        Array of points of shape (N, 5/6) where 5/6 = x,y,z + point type
        + particle index [+ start (0) or end (1) point tagging]
    """
    # Check on dimension
    if dim not in [2, 3]:
        raise ValueError("The image dimension must be either 2 or 3, "
                        f"got {dim} instead.")

    # Loop over true particles
    part_info = []
    for part_index, particle in enumerate(particle_v):
        # Check that the particle has the expected index
        assert part_index == particle.id()

        # If the particle does not meet minimum energy/size requirements, skip
        if (particle.energy_deposit() < min_energy_deposit or
            particle.num_voxels() < min_voxel_count):
            continue

        # If the particle is a nucleus, skip.
        # TODO: check if it's useful
        pdg_code = abs(particle.pdg_code())
        if pdg_code > 1000000000:  # Skipping nucleus trackid
            continue

        # If a shower has its first step outside of detector boundaries, skip
        # TODO: check if it's useful
        if pdg_code == 11 or pdg_code == 22:
            if not image_contains(meta, particle.first_step(), dim):
                continue

        # Skip low energy scatters and unknown shapes
        shape = particle.shape()
        if particle.shape() in [LOWES_SHP, UNKWN_SHP]:
            continue

        # Append the start point with the rest of the particle information
        first_step = image_coordinates(meta, particle.first_step(), dim)
        part_extra = [shape, part_index, 0] \
                if include_point_tagging else [shape, part_index]
        part_info.append(first_step + part_extra)

        # Append the end point as well, for tracks only
        if shape == TRACK_SHP:
            last_step  = image_coordinates(meta, particle.last_step(), dim)
            part_extra = [shape, part_index, 1] \
                    if include_point_tagging else [shape, part_index]
            part_info.append(last_step + part_extra)

    if not len(part_info):
        return np.empty((0, 5 + include_point_tagging), dtype=np.float32)

    return np.array(part_info)


def get_ppn_predictions(ppn_points, ppn_coords, ppn_masks,
                        ppn_classify_endpoints=None, segmentation=None,
                        ghost=None, score_threshold=0.5,
                        type_score_threshold=0.5, type_dist_threshold=1.999,
                        entry=None, pool_score_fn='max', pool_dist=1.999,
                        enforce_type=True, selection=None, num_classes=5, 
                        apply_deghosting=False):
    """Converts the raw output of PPN to a discrete set of proposed points.

    Parameters
    ----------
    ppn_points : Union[TensorBatch, List[np.ndarray]
         Raw output of PPN
    ppn_coords : Union[List[TensorBatch], List[List[np.ndarray]]
         Coordinates of the image at each PPN layer
    ppn_masks : Union[List[TensorBatch], List[List[np.ndarray]]
         Predicted masks of at each PPN layer
    ppn_classify_endpoints : Union[TensorBatch, List[np.ndarray]], optional
         Raw logits from the end point classification layer of PPN
    segmentation : Union[TensorBatch, List[np.ndarray]], optional
         Raw logits from the semantic segmentation network output
    ghost : Union[TensorBatch, List[np.ndarray]], optional
         Raw logits from the ghost segmentation network output
    score_threshold : float, default 0.5
         Score above which a point is considered to be active
    type_score_threshold : float, default 0.5
         Score above which a type prediction must be to be considered
    type_dist_threshold : float, default 1.999
         Distance threshold for matching with semantic type predictions
    entry : int, optional
         Entry in the batch for which to compute the point predictions
    pool_score_fn : str, default 'max'
         Which operation to use to pool PPN points scores ('max' or 'mean')
    pool_dist : float, default 1.999
         Distance below which PPN points should be merged into one (DBSCAN)
    enforce_type : bool, default True
         Whether to force PPN points predicted of type X to be within N voxels
         of a voxel with same predicted semantic type
    selection : List[List[int]], optional
         List of list of indices to consider exclusively (eg to get PPN 
         redictions within a cluster)
    num_classes : int, default 5
         Number of semantic classes
    apply_deghosting : bool, default False
         Whether to deghost the input, if a `ghost` tensor is provided

    Returns
    -------
    Union[np.ndarry, torch.Tensor]
        (N, P) Tensor of predicted points with P divided between
        [batch_id, x, y, z, validity scores (2), occupancy, type scores (5),
         predicted type, endpoint type]
    """
    # Fetch the segmentation tensor, if needed
    if enforce_type:
        assert segmentation is not None, (
                "Must provide the segmentation tensor to enforce types")
        if ghost is not None and apply_deghosting:
            mask_ghost = np.where(np.argmax(ghost, axis=1) == 0)[0]
            dist_segmentation = segmentation[mask_ghost]

    # Process the score pooling function
    if pool_score_fn == 'max':
        pool_fn = lambda x: np.max(x, axis=0)
    elif poof_score_fn == 'mean':
        pool_fn = lambda x: np.mean(x, axis=0)
    else:
        raise ValueError(
                f"Score pooling function not recognized: {score_pool}. "
                 "Should be one of 'max' or 'mean'")

    # Set the list of entries to loop over
    entries = range(len(ppn_points))
    if entry is not None:
        assert isinstance(entry, int), "If entry is specified, must be integer"
        entries = [entry]

    #  Loop over the entries in the batch
    classify_ends = ppn_classify_endpoints is not None
    output_list = []
    for b in entries:
        # Narrow down input to a specific entry
        ppn_raw_b = ppn_points[b]
        if isinstance(ppn_points, TensorBatch):
            ppn_mask_b = ppn_masks[-1][b].astype(bool).flatten()
            ppn_coords_b = ppn_coords[-1][b][:, COORD_COLS]
        else:
            ppn_mask_b = ppn_masks[b][-1].astype(bool).flatten()
            ppn_coords_b = ppn_coords[b][-1][:, COORD_COLS]
        if classify_ends:
            ppn_ends_b = ppn_classify_endpoints[b]

        # Restrict the PPN output to positive points above the score threshold
        scores = softmax(ppn_raw_b[:, PPN_RPOS_COLS], axis=1)
        mask = ppn_mask_b & (scores[:, -1] > score_threshold)

        # Restrict the PPN output to a subset of points, if requested
        if selection is not None:
            mask_update = np.zeros(mask.shape, dtype=bool)
            if entry is not None:
                assert (len(selection) == len(ppn_points) and
                        not np.issclar(selection[0]))
                mask_update[selection[b]] = True
            else:
                assert len(selection) and np.issclar(selection[0])
                mask_update[selection] = True

            mask &= mask_update

        # Apply the mask
        mask = np.where(mask)[0]
        scores = scores[mask]
        ppn_raw_b = ppn_raw_b[mask]
        ppn_coords_b = ppn_coords_b[mask]
        if classify_ends:
            ppn_ends_b = ppn_ends_b[mask]

        # Get the type predictions
        type_scores = softmax(ppn_raw_b[:, PPN_RTYPE_COLS], axis=1)
        type_pred = np.argmax(type_scores, axis=1)
        if classify_ends:
            end_scores = softmax(ppn_ends_b, axis=1)

        # Get the PPN point predictions
        coords = ppn_coords_b + 0.5 + ppn_raw_b[:, PPN_ROFF_COLS]
        if enforce_type:
            # Loop over the invidual classes
            seg_masks = []
            for c in range(num_classes):
                # Restrict the points to a specific class
                seg_pred = np.argmax(segmentation[b][mask], axis=1)
                seg_mask = seg_pred == c
                seg_mask &= type_scores[:, c] > type_score_threshold
                seg_mask = np.where(seg_mask)[0]

                # Make sure the points are within range of compatible class
                dist_mat = cdist(coords[seg_mask], ppn_coords_b[seg_mask])
                dist_mask = (dist_mat < type_dist_threshold).any(axis=1)
                seg_mask = seg_mask[dist_mask]

                seg_masks.append(seg_mask)

            # Restrict the available points further
            seg_mask = np.concatenate(seg_masks)

            coords = coords[seg_mask]
            scores = scores[seg_mask]
            type_pred = type_pred[seg_mask]
            type_scores = type_scores[seg_mask]
            end_scores = end_scores[seg_mask]

        # At this point, if there are no valid proposed points left, abort
        if not len(coords):
            output_list.append(
                    np.empty((0, 13 + 2*classify_ends), dtype=np.float32))

        # Cluster nearby points together
        clusts = dbscan_points(coords, eps=pool_dist, min_samples=1)
        output_list_b = []
        for c in clusts:
            types, cnts = np.unique(type_pred[c], return_counts=True)
            type_c = types[np.argmax(cnts)]
            output = [b, np.mean(coords[c], axis=0), pool_fn(scores[c]), 
                      len(c), pool_fn(type_scores[c]), type_c]
            if classify_ends:
                output.append(pool_fn(end_scores[c]))

            output_list_b.append(np.hstack(output))

        output_list.append(np.vstack(output_list_b))

    if entry is not None:
        return output_list[0]
    else:
        return TensorBatch.from_list(output_list)


def get_particle_points(coords, clusts, clusts_seg, ppn_points, classes=None,
                        anchor_points=True, enhance_track_points=False,
                        approx_farthest_points=True):
    """Associate PPN points with particle clusters.

    Given a list particle or fragment clusters, leverage the raw PPN output
    to produce a list of start points for shower objects and of end points
    for track objects:
    - For showers, pick the most likely PPN point
    - For tracks, pick the two points furthest away from each other

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates of voxels in the image
    clusts : List[numpy.ndarray]
        List of clusters representing the fragment or particle objects
    clusts_seg : numpy.ndarray
        Array of cluster semantic types
    ppn_points : numpy.ndarray
        Raw output of PPN
    anchor_points : bool, default True
        If `True`, the point estimates are brought to the closest cluster voxel
    approx_farthest_points: bool, default True
        If `True`, approximate the computation of the two farthest points
    enhance_track_points, default False
        If `True`, tracks leverage PPN predictions to provide a more
        accurate estimate of the end points. This needs to be avoided for
        track fragments, as PPN is typically not trained to find end points
        for them. If set to `False`, the two voxels furthest away from each
        other are picked.
    """

    # Loop over the relevant clusters
    points = np.empty((len(clusts), 6), dtype=np.float32)
    for i, c in enumerate(clusts):
        # Get cluster coordinates
        clust_coords = coords[c]

        # Deal with tracks
        if clusts_seg[i] == TRACK_SHP:
            # Get the two most separated points in the cluster
            idxs = [0, 0]
            method = 'brute' if not approx_farthest_points else 'recursive'
            idxs[0], idxs[1], _ = nbl.farthest_pair(clust_coords, method)
            end_points = clust_coords[idxs]

            # If requested, enhance using the PPN predictions. Only consider
            # points in the cluster that have a positive score
            if enhance_track_points:
                pos_mask = (ppn_points[c][idxs, PPN_RPOS_COLS[1]] >=
                            ppn_points[c][idxs, PPN_RPOS_COLS[0]])
                end_points += pos_mask * (points_tensor[idxs, :3] + 0.5)

            # If needed, anchor the track endpoints to the track cluster
            if anchor_points and enhance_track_points:
                dist_mat   = nbl.cdist(end_points, clust_coords)
                end_points = clust_coords[np.argmin(dist_mat, axis=1)]

            # Store
            points[i] = end_points.flatten()

        # Deal with the rest (EM activity)
        else:
            # Only use positive voxels and give precedence to predictions
            # that are contained within the voxel making the prediction.
            ppn_scores = softmax(ppn_points[c][:, PPN_RPOS_COLS], axis=1)[:,-1]
            val_index  = np.where(np.all(np.abs(ppn_points[c, :3] < 1.)))[0]
            best_id    = val_index[np.argmax(ppn_scores[val_index])] \
                    if len(val_index) else np.argmax(ppn_scores)
            start_point = clust_coords[best_id] \
                    + ppn_points[c][best_id, :3] + 0.5

            # If needed, anchor the shower start point to the shower cluster
            if anchor_points:
                dists = nbl.cdist(np.atleast_2d(start_point), clust_coords)
                start_point = clust_coords[np.argmin(dists)]

            # Store twice to preserve the feature vector length
            points[i] = np.concatenate([start_point, start_point])

    # Return points
    return points


def check_track_orientation_ppn(start_point, end_point, ppn_candidates):
    """Use PPN end point predictions to predict track orientation.

    Use the PPN point assignments as a basis to orient a track. Match
    the end points of a track to the closest PPN candidate and pick the
    candidate with the highest start score as the start point

    Parameters
    ----------
    start_point : np.ndarray
        (3) Start point of the track
    end_point : np.ndarray
        (3) End point of the track
    ppn_candidates : np.ndarray
        (N, 10)  PPN point candidates and their associated scores

    Returns
    -------
    bool
       Returns `True` if the start point provided is correct, `False`
       if the end point is more likely to be the start point.
    """
    # If there's no PPN candidates, nothing to do here
    if not len(ppn_candidates):
        return True

    # Get the candidate coordinates and end point classification predictions
    ppn_points = ppn_candidates[:, COORD_COLS]
    end_scores = ppn_candidates[:, PPN_END_COLS]

    # Compute the distance between the track end points and the PPN candidates
    end_points = np.vstack([start_point, end_point])
    dist_mat = nbl.cdist(end_points, ppn_points)

    # If both track end points are closest to the same PPN point, the start
    # point must be closest to it if the score is high, farthest otherwise
    argmins = np.argmin(dist_mat, axis=1)
    if argmins[0] == argmins[1]:
        label = np.argmax(end_scores[argmins[0]])
        dists = dist_mat[[0,1], argmins]
        return ((label == 0 and dists[0] < dists[1]) or
                (label == 1 and dists[1] < dists[0]))

    # In all other cases, check that the start point is associated with the PPN
    # point with the lowest end score
    end_scores = end_scores[argmins, -1]
    return end_scores[0] < end_scores[1]


def image_contains(meta, point, dim=3):
    """Checks whether a point is contained in the image box defined by meta.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to check on
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    """
    if dim == 3:
        return (point.x() >= meta.min_x() and point.y() >= meta.min_y() and 
                point.z() >= meta.min_z() and point.x() <= meta.max_x() and
                point.y() <= meta.max_y() and point.z() <= meta.max_z())
    else:
        return (point.x() >= meta.min_x() and point.x() <= meta.max_x() and
                point.y() >= meta.min_y() and point.y() <= meta.max_y())


def image_coordinates(meta, point, dim=3):
    """Returns the coordinates of a point in units of pixels with an image.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to convert the units of
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    """
    x, y, z = point.x(), point.y(), point.z()
    if dim == 3:
        x = (x - meta.min_x()) / meta.size_voxel_x()
        y = (y - meta.min_y()) / meta.size_voxel_y()
        z = (z - meta.min_z()) / meta.size_voxel_z()
        return [x, y, z]
    else:
        x = (x - meta.min_x()) / meta.size_voxel_x()
        y = (y - meta.min_y()) / meta.size_voxel_y()
        return [x, y]
