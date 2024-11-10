"""Module which contains utility function to process PPN information.

It contains functions to produce PPN labels and functions to process the
PPN predictions into something human-readable.
"""

import numpy as np
import torch
from warnings import warn
from typing import Union, List

from scipy.special import softmax as softmax_sp
from scipy.spatial.distance import cdist as cdist_sp

from spine.data import TensorBatch

from . import numba_local as nbl
from .dbscan import dbscan_points
from .torch_local import local_cdist
from .globals import (
        BATCH_COL, COORD_COLS, PPN_ROFF_COLS, PPN_RTYPE_COLS, PPN_RPOS_COLS,
        PPN_SCORE_COLS, PPN_OCC_COL, PPN_CLASS_COLS, PPN_SHAPE_COL,
        PPN_END_COLS, SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,
        UNKWN_SHP)


class PPNPredictor:
    """PPN post-processing class to convert PPN raw predictions into points."""

    def __init__(self, score_threshold=0.5, type_score_threshold=0.5,
                 type_dist_threshold=1.999, pool_score_fn='max',
                 pool_dist=1.999, enforce_type=True,
                 classes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                 apply_deghosting=False):
        """Initialize the PPN post-processor.

        Parameters
        ----------
        score_threshold : float, default 0.5
             Score above which a point is considered to be active
        type_score_threshold : float, default 0.5
             Score above which a type prediction must be to be considered
        type_dist_threshold : float, default 1.999
             Distance threshold for matching with semantic type predictions
        pool_score_fn : str, default 'max'
             Which operation to use to pool PPN points scores ('max' or 'mean')
        pool_dist : float, default 1.999
             Distance below which PPN points should be merged into one (DBSCAN)
        enforce_type : bool, default True
             Whether to force PPN points predicted of type X to be within N
             voxels of a voxel with same predicted semantic type
        classes : List[int], default [0, 1, 2, 3]
             Number of semantic classes
        apply_deghosting : bool, default False
             Whether to deghost the input, if a `ghost` tensor is provided
        """
        # Store the parameters
        self.score_threshold = score_threshold
        self.type_score_threshold = type_score_threshold
        self.type_dist_threshold = type_dist_threshold
        self.enforce_type = enforce_type
        self.classes = classes
        self.apply_deghosting = apply_deghosting

        # Store the score pooling function
        self.pool_dist = pool_dist
        self.pool_score_fn = pool_score_fn

    def __call__(self, ppn_points, ppn_coords, ppn_masks,
                 ppn_classify_endpoints=None, segmentation=None, ghost=None,
                 entry=None, selection=None, **kwargs):
        """Converts the batched raw output of PPN to a discrete set of
        proposed points of interest.

        Notes
        -----
        This function works on both wrapped (:class:`TensorBatch`) and
        unwrapped (`List[np.ndarray]`) batches of data.

        Parameters
        ----------
        ppn_points : Union[TensorBatch, List[np.ndarray]]
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
        entry : int, optional
             Entry in the batch for which to compute the point predictions
        selection : Union[IndexBatch, List[np.ndarray]], optional
             List of indexes to consider exclusively (e.g. to get PPN
             predictions within a list of clusters)
        **kwargs : dict, optional
             Extraneous outputs not used in this post-processor

        Returns
        -------
        Union[TensorBatch, List[np.ndarray]]
            (N, P) Tensor of predicted points with P divided between
            [batch_id, x, y, z, validity scores (2), occupancy, type scores (5),
             predicted type, endpoint type]
        """
        # Set the list of entries to loop over
        if entry is not None:
            assert isinstance(entry, int), (
                    "If entry is specified, must be integer")
            entries = [entry]
        else:
            entries = range(len(ppn_points))

        # Loop over the entries, process it
        ppn_pred = []
        ppn_classify_endpoints_b, segmentation_b, ghost_b, selection_b = (
                None, None, None, None)
        for b in range(len(ppn_points)):
            # Prepare input for that entry
            ppn_points_b = ppn_points[b]
            if isinstance(ppn_points, TensorBatch):
                ppn_coords_b = ppn_coords[-1][b][:, COORD_COLS]
                ppn_mask_b = ppn_masks[-1][b].flatten()
            else:
                ppn_coords_b = ppn_coords[b][-1][:, COORD_COLS]
                ppn_mask_b = ppn_masks[b][-1].flatten()
            if ppn_classify_endpoints is not None:
                ppn_classify_endpoints_b = ppn_classify_endpoints[b]
            if segmentation is not None:
                segmentation_b = segmentation[b]
            if ghost is not None:
                ghost_b = ghost[b]
            if selection is not None:
                selection_b = selection[b]

            # Append
            self.entry = b
            ppn_pred.append(self.process_single(
                ppn_points_b, ppn_coords_b, ppn_mask_b,
                ppn_classify_endpoints_b, segmentation_b, ghost_b,
                selection_b))

        # Return
        if entry is not None:
            return ppn_pred[0]
        elif not isinstance(ppn_points, TensorBatch):
            return ppn_pred
        else:
            tensor = TensorBatch.from_list(ppn_pred)
            tensor.coord_cols = COORD_COLS
            return tensor

    def process_single(self, ppn_raw, ppn_coords, ppn_mask, ppn_ends=None,
                       segmentation=None, ghost=None, selection=None):
        """Converts the PPN output from a single entry into points of interests
        for that entry.

        Notes
        -----
        This function works both `torch.Tensor` and `np.ndarray` objects.

        Parameters
        ----------
        ppn_raw : Union[torch.Tensor, np.ndarray]
             Raw output of PPN
        ppn_coords : Union[torch.Tensor, np.ndarray]
             Coordinates of the image at each PPN layer
        ppn_masks : Union[torch.Tensor, np.ndarray]
             Predicted masks of at each PPN layer
        ppn_ends : Union[torch.Tensor, np.ndarray], optional
             Raw logits from the end point classification layer of PPN
        segmentation : Union[torch.Tensor, np.ndarray], optional
             Raw logits from the semantic segmentation network output
        ghost : Union[torch.Tensor, np.ndarray], optional
             Raw logits from the ghost segmentation network output
        selection : Union[torch.Tensor, np.ndarray], optional
             List of indexes to consider exclusively (e.g. to get PPN
             predictions within a list of clusters)

        Returns
        -------
        Union[TensorBatch, List[np.ndarray]]
            (N, P) Tensor of predicted points with P divided between
            [batch_id, x, y, z, validity scores (2), occupancy, type scores (5),
             predicted type, endpoint type]
        """
        # Define operations on the basis of the input type
        if torch.is_tensor(ppn_raw):
            dtype, device = ppn_raw.dtype, ppn_raw.device
            cat, unique, argmax = torch.cat, torch.unique, torch.argmax
            where, mean, softmax = torch.where, torch.mean, torch.softmax
            cdist = local_cdist
            empty = lambda x: torch.empty(x, dtype=dtype, device=device)
            zeros = lambda x: torch.zeros(x, dtype=dtype, device=device)
            pool_fn = getattr(torch, self.pool_score_fn)
            if self.pool_score_fn == 'max':
                pool_fn = torch.amax

        else:
            cat, unique, argmax = np.concatenate, np.unique, np.argmax
            where, mean, softmax = np.where, np.mean, softmax_sp
            cdist = cdist_sp
            empty = lambda x: np.empty(x, dtype=ppn_raw.dtype)
            zeros = lambda x: np.zeros(x, dtype=ppn_raw.dtype)
            pool_fn = getattr(np, self.pool_score_fn)

        # Fetch the segmentation tensor, if needed
        if self.enforce_type:
            assert segmentation is not None, (
                    "Must provide the segmentation tensor to enforce types")
            if ghost is not None and self.apply_deghosting:
                mask_ghost = where(argmax(ghost, 1) == 0)[0]
                segmentation = segmentation[mask_ghost]

        # Restrict the PPN output to points above the score threshold
        scores = softmax(ppn_raw[:, PPN_RPOS_COLS], 1)
        mask = ppn_mask & (scores[:, -1] > self.score_threshold)

        # Restrict the PPN output to a subset of points, if requested
        if selection is not None:
            mask_update = zeros(mask.shape, dtype=bool)
            if entry is not None:
                assert (len(selection) == len(ppn_points) and
                        not np.issclar(selection[0]))
                mask_update[selection[b]] = True
            else:
                assert len(selection) and np.issclar(selection[0])
                mask_update[selection] = True

            mask &= mask_update

        # Apply the mask
        mask = where(mask)[0]
        scores = scores[mask]
        ppn_raw = ppn_raw[mask]
        ppn_coords = ppn_coords[mask]
        if ppn_ends is not None:
            ppn_ends = ppn_ends[mask]

        # Get the type predictions
        type_scores = softmax(ppn_raw[:, PPN_RTYPE_COLS], 1)
        type_pred = argmax(type_scores, 1)
        if ppn_ends is not None:
            end_scores = softmax(ppn_ends, 1)

        # Get the PPN point predictions
        coords = ppn_coords + 0.5 + ppn_raw[:, PPN_ROFF_COLS]
        if self.enforce_type:
            # Loop over the invidual classes
            seg_masks = []
            for c in self.classes:
                # Restrict the points to a specific class
                seg_pred = argmax(segmentation[mask], 1)
                seg_mask = seg_pred == c
                seg_mask &= type_scores[:, c] > self.type_score_threshold
                seg_mask = where(seg_mask)[0]

                # Make sure the points are within range of compatible class
                dist_mat = cdist(coords[seg_mask], ppn_coords[seg_mask])
                dist_mask = (dist_mat < self.type_dist_threshold).any(1)
                seg_mask = seg_mask[dist_mask]

                seg_masks.append(seg_mask)

            # Restrict the available points further
            seg_mask = cat(seg_masks)

            coords = coords[seg_mask]
            scores = scores[seg_mask]
            type_pred = type_pred[seg_mask]
            type_scores = type_scores[seg_mask]
            if ppn_ends is not None:
                end_scores = end_scores[seg_mask]

        # At this point, if there are no valid proposed points left, abort
        if not len(coords):
            return empty((0, 13 + 2*(ppn_ends is not None)))

        # Cluster nearby points together
        if torch.is_tensor(coords):
            clusts = dbscan_points(
                    coords.detach().cpu().numpy(), eps=self.pool_dist,
                    min_samples=1)
        else:
            clusts = dbscan_points(coords, eps=self.pool_dist, min_samples=1)

        ppn_pred = empty((len(clusts), 13 + 2*(ppn_ends is not None)))
        for i, c in enumerate(clusts):
            types, cnts = unique(type_pred[c], return_counts=True)
            type_c = types[argmax(cnts)]
            ppn_pred[i, BATCH_COL] = self.entry
            ppn_pred[i, COORD_COLS] = mean(coords[c], 0)
            ppn_pred[i, PPN_SCORE_COLS] = pool_fn(scores[c], 0)
            ppn_pred[i, PPN_OCC_COL] = len(c)
            ppn_pred[i, PPN_CLASS_COLS] = pool_fn(type_scores[c], 0)
            ppn_pred[i, PPN_SHAPE_COL] = type_c
            if ppn_ends is not None:
                ppn_pred[i, PPN_END_COLS] = pool_fn(end_scores[c], 0)

        return ppn_pred


def get_particle_points(data, clusts, clusts_seg, ppn_points,
                        anchor_points=True, enhance_track_points=False,
                        approx_farthest_points=True):
    """Associate PPN points with particle clusters.

    Given a list particle or fragment clusters, leverage the raw PPN output
    to produce a list of start points for shower objects and of start/end
    points for track objects:
    - For showers, pick the most likely PPN point
    - For tracks, pick the two points farthest away from each other

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
        for them. If set to `False`, the two voxels farthest away from each
        other are picked.
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(data.tensor):
        dtype, device = data.dtype, data.device
        cat, argmin, argmax = torch.cat, torch.argmin, torch.argmax
        where, abss, softmax = torch.where, torch.abs, torch.softmax
        cdist = local_cdist
        empty = lambda x: torch.empty(x, dtype=dtype, device=device)
        def farthest_pair(x):
            if len(x) < 2:
                return [0, 0]
            return list(torch.triu_indices(
                len(x), len(x), 1)[:, torch.argmax(torch.pdist(x))])

    else:
        cat, argmin, argmax = np.concatenate, np.argmin, np.argmax
        where, abss, softmax = np.where, np.abs, softmax_sp
        cdist = cdist_sp
        empty = lambda x: np.empty(x, dtype=data.dtype)
        farthest_pair = lambda x : list(nbl.farthest_pair(
                x, approx_farthest_points)[:2])

    # Loop over the relevant clusters
    points = empty((len(clusts.index_list), 6))
    for i, c in enumerate(clusts.index_list):
        # Get cluster coordinates
        clust_coords = data.tensor[c][:, COORD_COLS]
        points_tensor = ppn_points.tensor[c]

        # For tracks, find the two poins farthest away from each other
        if clusts_seg.tensor[i] == TRACK_SHP:
            # Get the two most separated points in the cluster
            idxs = farthest_pair(clust_coords)
            idxs[0], idxs[1] = int(idxs[0]), int(idxs[1])
            end_points = clust_coords[idxs]

            # If requested, enhance using the PPN predictions. Only consider
            # points in the cluster that have a positive score
            if enhance_track_points:
                pos_mask = (points_tensor[idxs, PPN_RPOS_COLS[1]] >=
                            points_tensor[idxs, PPN_RPOS_COLS[0]])
                end_points += pos_mask * (
                        points_tensor[idxs][:, PPN_ROFF_COLS] + 0.5)

                # If needed, anchor the track endpoints to the track cluster
                if anchor_points:
                    dist_mat   = cdist(end_points, clust_coords)
                    end_points = clust_coords[argmin(dist_mat, 1)]

            # Store
            points[i] = end_points.flatten()

        # For showers, find the most likely point
        else:
            # Only use positive voxels and give precedence to predictions
            # that are contained within the voxel making the prediction.
            ppn_scores = softmax(points_tensor[:, PPN_RPOS_COLS], 1)[:, -1]
            dists = abss(points_tensor[:, PPN_ROFF_COLS])

            val_index = where((ppn_scores > 0.5) & (dists < 1.).all(1))[0]
            if len(val_index):
                best_id = val_index[argmax(ppn_scores[val_index])]
            else:
                best_id = argmax(ppn_scores)

            start_point = (clust_coords[best_id]
                           + points_tensor[best_id, PPN_ROFF_COLS] + 0.5)

            # If needed, anchor the shower start point to the shower cluster
            if anchor_points:
                dists = cdist(start_point[None, :], clust_coords)
                start_point = clust_coords[argmin(dists)]

            # Store twice to preserve the feature vector length
            points[i] = cat([start_point, start_point], 0)

    # Return points
    return TensorBatch(points, clusts.counts, coord_cols=np.arange(6))


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


def get_ppn_labels(particle_v, meta, dtype, dim=3, min_voxel_count=1,
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
    dtype : str
        Typing of the output PPN labels
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
        if part_index != particle.id():
            warn("Particle list index does not match its `id` attribute.")

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
        return np.empty((0, 5 + include_point_tagging), dtype=dtype)

    return np.array(part_info, dtype=dtype)


def get_vertex_labels(particle_v, neutrino_v, meta, dtype):
    """Gets particle vertex coordinates.

    It provides the coordinates of points where multiple particles originate:
    - If the `neutrino_event` is provided, it simply uses the coordinates of
      the neutrino interaction points.
    - If the `particle_event` is provided instead, it looks for ancestor point
      positions shared by at least two **primary** particles.

    Parameters
    ----------
    particle_v : List[larcv.Particle]
        List of LArCV particle objects in the image
    neutrino_v : List[larcv.Neutrino]
        List of LArCV neutrino objects in the image
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    dtype : str
        Typing of the output PPN labels

    Returns
    -------
    np.array
        Array of points of shape (N, 4) where 4 = x, y, z, vertex_id
    """
    # If the particles are provided, find unique ancestors
    vertexes = []
    if particle_v is not None:
        # Fetch all ancestor positions of primary particles
        anc_positions = []
        for i, p in enumerate(particle_v):
            if p.parent_id() == p.id() or p.ancestor_pdg_code() == 111:
                if image_contains(meta, p.ancestor_position()):
                    anc_pos = image_coordinates(meta, p.ancestor_position())
                    anc_positions.append(anc_pos)

        # If there is no primary, nothing to do
        if not len(anc_positions):
            return np.empty((0, 4), dtype=dtype)

        # Find those that appear > once
        anc_positions = np.vstack(anc_positions)
        unique_positions, counts = np.unique(
                anc_positions, return_counts=True, axis=0)
        for i, idx in enumerate(np.where(counts > 1)[0]):
            vertexes.append([*unique_positions[idx], i])

    # If the neutrinos are provided, straightforward
    if neutrino_v is not None:
        for i, n in enumerate(neutrino_v):
            if image_contains(meta, n.position()):
                nu_pos = image_coordinates(meta, n.position())
                vertexes.append([*nu_pos, i])

    # If there are no vertex, nothing to do
    if not len(vertexes):
        return np.empty((0, 4), dtype=dtype)

    return np.vstack(vertexes).astype(dtype)


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
