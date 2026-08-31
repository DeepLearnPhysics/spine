"""Module which contains utility function to process PPN information.

It contains functions to produce PPN labels and functions to process the
PPN predictions into something human-readable.
"""

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.constants import (
    DELTA_SHP,
    MICHL_SHP,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.data import IndexBatch, TensorBatch, TensorData, TensorSchema

from .conditional import torch
from .jit import numbafy
from .torch.runtime import cdist_fast


def ppn_raw_schema() -> TensorSchema:
    """Return the schema of the raw PPN prediction head output."""
    return TensorSchema(
        feature_fields={
            "offsets": (0, 1, 2),
            "type_logits": (3, 4, 5, 6, 7),
            "point_logits": (8, 9),
        },
        feats_only=True,
    )


def ppn_prediction_schema(endpoints: bool = False) -> TensorSchema:
    """Return the schema of discrete post-processed point predictions.

    Parameters
    ----------
    endpoints : bool, default False
        Include the two endpoint-score columns used for track orientation.
    """
    fields = {
        "scores": (0, 1),
        "occupancy": (2,),
        "class_scores": (3, 4, 5, 6, 7),
        "shape": (8,),
    }
    if endpoints:
        fields["endpoint_scores"] = (9, 10)
    return TensorSchema(
        coordinate_groups={"point": (0, 1, 2)},
        feature_fields=fields,
    )


class PPNPredictor:
    """PPN post-processing class to convert PPN raw predictions into points."""

    def __init__(
        self,
        score_threshold: float = 0.5,
        type_score_threshold: float = 0.5,
        type_dist_threshold: float = 1.999,
        pool_score_fn: str = "max",
        pool_dist: float = 1.999,
        enforce_type: bool = True,
        classes: Sequence[int] = (SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP),
        apply_deghosting: bool = False,
    ) -> None:
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
        self.classes = tuple(classes)
        self.apply_deghosting = apply_deghosting

        # Store the score pooling function
        self.pool_dist = pool_dist
        self.pool_score_fn = pool_score_fn

    def __call__(
        self,
        ppn_points,
        ppn_coords,
        ppn_masks,
        ppn_classify_endpoints=None,
        segmentation=None,
        ghost=None,
        entry=None,
        selection=None,
        **kwargs,
    ):
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
        # Wrapped and unwrapped batches share the same event-level contract;
        # only their outer containers differ.
        is_batched = isinstance(ppn_points, TensorBatch)

        # Set the list of entries to loop over
        if entry is not None:
            if not isinstance(entry, int):
                raise TypeError("If `entry` is specified, it must be an integer.")
            entries = [entry]
        else:
            entries = range(len(ppn_points))

        # Loop over the entries, process it
        ppn_pred: list[TensorData] = []
        ppn_classify_endpoints_b, segmentation_b, ghost_b, selection_b = (
            None,
            None,
            None,
            None,
        )
        for b in entries:
            # Prepare input for that entry
            if is_batched:
                ppn_points_b = ppn_points.event(b)
                ppn_coords_b = ppn_coords[-1].coords[b]
                ppn_mask_b = ppn_masks[-1].features[b].flatten()
            else:
                ppn_points_b = ppn_points[b]
                ppn_coords_b = ppn_coords[b][-1].coords
                ppn_mask_b = ppn_masks[b][-1].features.flatten()
            if ppn_classify_endpoints is not None:
                endpoint = ppn_classify_endpoints[b]
                ppn_classify_endpoints_b = endpoint if is_batched else endpoint.features
            if segmentation is not None:
                segment = segmentation[b]
                segmentation_b = segment if is_batched else segment.features
            if ghost is not None:
                ghost_entry = ghost[b]
                ghost_b = ghost_entry if is_batched else ghost_entry.features
            if selection is not None:
                selection_b = selection[b]

            # Append
            ppn_pred.append(
                self.process_single(
                    ppn_points_b,
                    ppn_coords_b,
                    ppn_mask_b,
                    ppn_classify_endpoints_b,
                    segmentation_b,
                    ghost_b,
                    selection_b,
                )
            )

        # Return
        if entry is not None:
            return ppn_pred[0]
        elif not is_batched:
            return ppn_pred
        else:
            return TensorBatch.from_data_list(ppn_pred)

    def process_single(
        self,
        ppn_raw: TensorData,
        ppn_coords,
        ppn_mask,
        ppn_ends=None,
        segmentation=None,
        ghost=None,
        selection=None,
    ) -> TensorData:
        """Converts the PPN output from a single entry into points of interests
        for that entry.

        Notes
        -----
        This function works with both `torch.Tensor` and `np.ndarray` inputs.

        Parameters
        ----------
        ppn_raw : TensorData
             Raw output of PPN with named feature fields
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
        TensorData
            Predicted points with named coordinates and feature fields.
        """
        raw_features = ppn_raw.features
        offsets = ppn_raw.feature("offsets")
        type_logits = ppn_raw.feature("type_logits")
        point_logits = ppn_raw.feature("point_logits")

        # Define operations on the basis of the input type
        if torch.is_tensor(raw_features):
            dtype, device = raw_features.dtype, raw_features.device
            cat, unique, argmax = torch.cat, torch.unique, torch.argmax
            where, mean, softmax = torch.where, torch.mean, torch.softmax
            cdist = cdist_fast

            def empty(size):
                return torch.empty(size, dtype=dtype, device=device)

            def zeros_bool(size):
                return torch.zeros(size, dtype=torch.bool, device=device)

            pool_fn = getattr(torch, self.pool_score_fn)
            if self.pool_score_fn == "max":
                pool_fn = torch.amax

        else:
            cat, unique, argmax = np.concatenate, np.unique, np.argmax
            where, mean = np.where, np.mean
            softmax, cdist = sm.softmax, sm.distance.cdist

            def empty(size):
                return np.empty(size, dtype=raw_features.dtype)

            def zeros_bool(size):
                return np.zeros(size, dtype=bool)

            pool_fn = getattr(np, self.pool_score_fn)

        # Fetch the segmentation tensor, if needed
        segmentation_values = segmentation
        if self.enforce_type:
            if segmentation_values is None:
                raise ValueError("Must provide segmentation to enforce PPN types.")
            if ghost is not None and self.apply_deghosting:
                mask_ghost = where(argmax(ghost, 1) == 0)[0]
                segmentation_values = segmentation_values[mask_ghost]

        # Restrict the PPN output to points above the score threshold
        scores = softmax(point_logits, 1)
        mask = ppn_mask & (scores[:, -1] > self.score_threshold)

        # Restrict the PPN output to a subset of points, if requested
        if selection is not None:
            mask_update = zeros_bool(mask.shape)
            mask_update[selection] = True

            mask &= mask_update

        # Apply the mask
        mask = where(mask)[0]
        scores = scores[mask]
        offsets = offsets[mask]
        type_logits = type_logits[mask]
        ppn_coords = ppn_coords[mask]
        if ppn_ends is not None:
            ppn_ends = ppn_ends[mask]

        # Get the type predictions
        type_scores = softmax(type_logits, 1)
        type_pred = argmax(type_scores, 1)
        end_scores = None
        if ppn_ends is not None:
            end_scores = softmax(ppn_ends, 1)

        # Get the PPN point predictions
        coords = ppn_coords + 0.5 + offsets
        if self.enforce_type:
            assert segmentation_values is not None
            # Loop over the invidual classes
            seg_masks = []
            for c in self.classes:
                # Restrict the points to a specific class
                seg_pred = argmax(segmentation_values[mask], 1)
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
            if end_scores is not None:
                end_scores = end_scores[seg_mask]

        # At this point, if there are no valid proposed points left, abort
        schema = ppn_prediction_schema(ppn_ends is not None)
        if not len(coords):
            return TensorData(
                coords=empty((0, 3)),
                features=empty((0, 9 + 2 * (ppn_ends is not None))),
                schema=schema,
            )

        # Cluster nearby points together
        if torch.is_tensor(coords):
            clusts = self.dbscan_points(coords.detach().cpu().numpy())
        else:
            clusts = self.dbscan_points(coords)

        point_coords = empty((len(clusts), 3))
        point_features = empty((len(clusts), 9 + 2 * (ppn_ends is not None)))
        fields = schema.feature_fields
        for i, c in enumerate(clusts):
            types, cnts = unique(type_pred[c], return_counts=True)
            type_c = types[argmax(cnts)]
            point_coords[i] = mean(coords[c], 0)
            point_features[i, fields["scores"]] = pool_fn(scores[c], 0)
            point_features[i, fields["occupancy"]] = len(c)
            point_features[i, fields["class_scores"]] = pool_fn(type_scores[c], 0)
            if torch.is_tensor(type_c):
                type_c = type_c.to(dtype=dtype)
            point_features[i, fields["shape"]] = type_c
            if end_scores is not None:
                point_features[i, fields["endpoint_scores"]] = pool_fn(end_scores[c], 0)

        return TensorData(coords=point_coords, features=point_features, schema=schema)

    def dbscan_points(self, coordinates):
        """Form clusters of predited points based on proximity.

        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates of the points to cluster

        Returns
        -------
        List[np.ndarray]
            List of proposed point cluster indexes
        """
        # Assign cluster labels to all proposed poins
        labels = sm.cluster.dbscan(coordinates, eps=self.pool_dist, min_samples=1)

        # Convert the list of labels into a list of cluster indexes
        clusts = []
        for c in np.unique(labels):
            clusts.append(np.where(labels == c)[0])

        return clusts


class ParticlePointPredictor:
    """Produces start/end points given a list of particles and PPN predictions.

    Given a list particle or fragment clusters, leverage the raw PPN output
    to produce a list of start points for shower objects and of start/end
    points for track objects:
    - For showers, pick the most likely PPN point
    - For tracks, pick the two points farthest away from each other
    """

    def __init__(
        self,
        use_numpy: bool = True,
        contained_first: bool = True,
        anchor_points: bool = True,
        enhance_track_points: bool = False,
        approx_farthest_points: bool = True,
    ) -> None:
        """Initialize the particle point predictor.

        Parameters
        ----------
        use_numpy : bool, default True
            Compute the particle start/end points on CPU using numpy+numba
        contained_first : bool, default True
            If `True`, for shower points, give precedence to voxels which
            predict a point within one voxel of their location
        anchor_points : bool, default True
            If `True`, the point estimates are brought to the closest cluster voxel
        enhance_track_points : bool, default False
            If `True`, tracks leverage PPN predictions to provide a more
            accurate estimate of the end points. This needs to be avoided for
            track fragments, as PPN is typically not trained to find end points
            for them. If set to `False`, the two voxels farthest away from each
            other are picked.
        approx_farthest_points: bool, default True
            If `True`, approximate the computation of the two farthest points
        """
        # Store the point predictor parameters
        self.use_numpy = use_numpy
        self.contained_first = contained_first
        self.anchor_points = anchor_points
        self.enhance_track_points = enhance_track_points
        self.approx_farthest_points = approx_farthest_points

    def __call__(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        clust_shapes: TensorBatch,
        ppn_points: TensorBatch,
    ) -> TensorBatch:
        """Assign start/end points to one batch of events.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, TensorBatch]
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        clusts : Union[List[np.ndarray], IndexBatch]
            List of particle clusters
        clust_shapes : Union[np.ndarray, torch.Tensor, TensorBatch]
            Semantic type of each of the clusters
        ppn_points : TensorBatch
            Raw output of PPN
        """
        if not clusts.is_list:
            raise TypeError(
                "Particle point prediction expects list-backed cluster indexes."
            )
        points = data.coords.tensor
        shapes = clust_shapes.tensor
        offsets = ppn_points.feature("offsets").tensor
        point_logits = ppn_points.feature("point_logits").tensor
        cluster_list, counts = clusts.index_list, clusts.counts

        # Get cluster point coordinates, dispatch
        if not self.use_numpy:
            # Check type, pass to torch function
            if not isinstance(points, torch.Tensor):
                raise TypeError("Torch point prediction requires torch-backed data.")
            end_points = self.get_end_points_torch(
                points, cluster_list, shapes, offsets, point_logits
            )

        else:
            # Pass to numpy function (takes care of object conversion)
            end_points = self.get_end_points_numpy(
                points, cluster_list, shapes, offsets, point_logits
            )

        return TensorBatch(
            end_points,
            counts,
            coord_cols=np.arange(6),
            schema=TensorSchema(
                coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)}
            ),
        )

    def get_end_points_torch(self, points, clusts, clusts_seg, offsets, point_logits):
        """Torch function to fetch each of the cluster end points.

        Parameters
        ----------
        points : torch.Tensor
            (N, 3) Image point coordinates
        clusts : List[np.ndarray]
            List of particle clusters
        clust_shapes : np.ndarray
            Semantic type of each of the clusters
        ppn_points : torch.Tensor
            Raw output of PPN
        """
        # Loop over the relevant clusters
        end_points = torch.empty(
            (len(clusts), 6), dtype=points.dtype, device=points.device
        )
        for i, c in enumerate(clusts):
            # Get cluster coordinates
            points_c = points[c]
            offsets_c = offsets[c]
            point_logits_c = point_logits[c]

            # For tracks, find the two poins farthest away from each other
            if clusts_seg[i] == TRACK_SHP:
                # Get the two most separated points in the cluster
                idx = torch.argmax(cdist_fast(points_c, points_c))
                idxs = sorted([int(idx // len(points_c)), int(idx % len(points_c))])
                track_points = points_c[idxs]

                # If requested, enhance using the PPN predictions. Only consider
                # points in the cluster that have a positive score
                if self.enhance_track_points:
                    pos_mask = point_logits_c[idxs, 1] >= point_logits_c[idxs, 0]
                    track_points += pos_mask[:, None] * (offsets_c[idxs] + 0.5)

                    # If needed, anchor the track endpoints to the track cluster
                    if self.anchor_points:
                        dist_mat = cdist_fast(track_points, points_c)
                        track_points = points_c[torch.argmin(dist_mat, 1)]

                # Store
                end_points[i] = track_points.flatten()

            # For showers, find the most likely point
            else:
                # Only use positive voxels and give precedence to predictions
                # that are contained within the voxel making the prediction.
                ppn_scores = torch.softmax(point_logits_c, 1)[:, -1]
                if self.contained_first:
                    dists = torch.abs(offsets_c)

                    val_index = torch.where((ppn_scores > 0.5) & (dists < 1.0).all(1))[
                        0
                    ]
                    if len(val_index):
                        best_id = val_index[torch.argmax(ppn_scores[val_index])]
                    else:
                        best_id = torch.argmax(ppn_scores)
                else:
                    best_id = torch.argmax(ppn_scores)

                start_point = points_c[best_id] + offsets_c[best_id] + 0.5

                # If needed, anchor the shower start point to the shower cluster
                if self.anchor_points:
                    dists = cdist_fast(start_point[None, :], points_c)
                    start_point = points_c[torch.argmin(dists)]

                # Store twice to preserve the feature vector length
                end_points[i] = torch.cat((start_point, start_point), 0)

        # Return points
        return end_points

    @numbafy(
        cast_args=["points", "offsets", "point_logits"],
        list_args=["clusts"],
        keep_torch=True,
        ref_arg="points",
    )
    def get_end_points_numpy(self, points, clusts, clust_shapes, offsets, point_logits):
        """Parallelized numba function to fetch each of the cluster end points.

        Parameters
        ----------
        points : np.darray
            (N, 3) Image point coordinates
        clusts : List[np.ndarray]
            List of particle clusters
        clust_shapes : np.darray
            Semantic type of each of the clusters
        ppn_points : np.ndarray
            Raw output of PPN
        """
        # If there are no clusters, nothing to do
        if len(clusts) == 0:
            return np.empty((0, 6), dtype=points.dtype)

        return self._get_end_points_numpy(
            points,
            clusts,
            clust_shapes,
            offsets,
            point_logits,
            self.contained_first,
            self.anchor_points,
            self.enhance_track_points,
            self.approx_farthest_points,
        )

    @staticmethod
    @nb.njit(cache=True, parallel=True, nogil=True)
    def _get_end_points_numpy(
        points: nb.float32[:, :],
        clusts: nb.types.List(nb.int64[:]),
        clust_shapes: nb.int64[:],
        offsets: nb.float32[:, :],
        point_logits: nb.float32[:, :],
        contained_first: nb.boolean,
        anchor_points: nb.boolean,
        enhance_track_points: nb.boolean,
        approx_farthest_pair: nb.boolean,
    ):
        # Loop over the relevant clusters
        end_points = np.empty((len(clusts), 6), dtype=points.dtype)
        for k in nb.prange(len(clusts)):
            # Get cluster coordinates
            c = clusts[k]
            points_c = points[c]
            offsets_c = offsets[c]
            point_logits_c = point_logits[c]

            # For tracks, find the two poins farthest away from each other
            if clust_shapes[k] == TRACK_SHP:
                # Get the two most separated points in the cluster
                idxs = np.sort(
                    np.array(
                        sm.distance.farthest_pair(points_c, approx_farthest_pair)[:2]
                    )
                )
                track_points = points_c[idxs]

                # If requested, enhance using the PPN predictions. Only consider
                # points in the cluster that have a positive score
                if enhance_track_points:
                    pos_mask = point_logits_c[idxs, 1] >= point_logits_c[idxs, 0]
                    track_points += pos_mask.reshape(-1, 1) * (
                        offsets_c[idxs] + np.array(0.5, dtype=points.dtype)
                    )

                    # If needed, anchor the track endpoints to the track cluster
                    if anchor_points:
                        dist_mat = sm.distance.cdist(track_points, points_c)
                        track_points = points_c[np.argmin(dist_mat, 1)]

                # Store
                end_points[k] = track_points.flatten()

            # For showers, find the most likely point
            else:
                # Only use positive voxels and give precedence to predictions
                # that are contained within the voxel making the prediction.
                ppn_scores = sm.softmax(point_logits_c, 1)[:, -1]
                if contained_first:
                    dists = np.abs(offsets_c)

                    val_index = np.where((ppn_scores > 0.5) & sm.all(dists < 1.0, 1))[0]
                    if len(val_index):
                        best_id = val_index[np.argmax(ppn_scores[val_index])]
                    else:
                        best_id = np.argmax(ppn_scores)
                else:
                    best_id = np.argmax(ppn_scores)

                start_point = (
                    points_c[best_id]
                    + offsets_c[best_id]
                    + np.array(0.5, dtype=points.dtype)
                )

                # If needed, anchor the shower start point to the shower cluster
                if anchor_points:
                    dists = sm.distance.cdist(start_point[None, :], points_c)
                    start_point = points_c[np.argmin(dists)]

                # Store twice to preserve the feature vector length
                end_points[k] = np.concatenate((start_point, start_point))

        return end_points
