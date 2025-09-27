"""Module which does connected-components (dense) clustering using DBSCAN."""

import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch
from spine.math.cluster import DBSCAN as spine_dbscan
from spine.utils.globals import (
    COORD_COLS,
    COORD_END_COLS,
    COORD_START_COLS,
    DELTA_SHP,
    MICHL_SHP,
    PPN_SHAPE_COL,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.utils.point_break_clustering import PointBreakClusterer
from spine.utils.ppn import PPNPredictor


class DBSCAN(torch.nn.Module):
    """Uses DBSCAN to find locally-dense particle fragments.

    It uses SPINE's numba-accelerated DBSCAN implementation to fragment each of
    the particle shapes into dense instances. Runs DBSCAN on each requested
    semantic class separately, in one of three ways:
    - Run pure DBSCAN on all the voxels in that class
    - Runs DBSCAN on PPN point-masked voxels and then associates the
      leftovers based on proximity to existing instances.
    - Use a graph-based method to cluster tracks based on PPN vertices. This
      technique can only be used on tracks.
    """

    def __init__(
        self,
        eps=1.8,
        min_samples=1,
        min_size=3,
        metric="euclidean",
        shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
        break_shapes=[TRACK_SHP],
        break_mask_radius=5.0,
        break_track_method="masked_dbscan",
        use_label_break_points=False,
        track_include_delta=False,
        ppn_predictor={},
    ):
        """Initialize the DBSCAN clustering algorithm.

        Parameters
        ----------
        eps : float, default 1.8
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
        min_samples : int, default 1
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point.
        min_size : int, default 3
            Minimum cluster size to stored in the final list of DBSCAN clusters
        metric : str, default 'euclidean'
            Metric used to compute the pair-wise distances between space points
        shapes : List[int], default [0, 1, 2, 3]
            List of semantic classes to run DBSCAN on
        break_shapes : List[int], default [1]
            List of semantic shapes for which to use PPN to break down
        break_mask_radius : str, default 5.0
            If using particle points to break up instances further, specifies
            the radius around each particle point which gets masked
        break_track_method : str, default 'masked_dbscan'
            If using particle points to break up tracks, specifies the method
        use_label_break_points : bool, default False
            Whether to use label points to break instances
        track_include_delta : bool, default False
            If `True`, include delta points along with track point when
            running DBSCAN on track points (limits artificial track breaks)
        ppn_predictor : cfg, optional
            PPN post-processing configuration
        """
        # Initialize the parent class
        super().__init__()

        # Store the DBSCAN clustering parameters
        self.eps = eps
        self.min_samples = min_samples
        self.min_size = min_size
        self.metric = metric
        self.shapes = shapes
        self.break_shapes = break_shapes
        self.break_mask_radius = break_mask_radius
        self.break_track_method = break_track_method
        self.track_include_delta = track_include_delta

        # If the constants are provided as scalars, turn them into lists
        assert not np.isscalar(shapes), "Semantic classes should be provided as a list."
        for attr in ["eps", "min_samples", "min_size", "metric", "break_mask_radius"]:
            if np.isscalar(getattr(self, attr)):
                setattr(self, attr, len(shapes) * [getattr(self, attr)])
            else:
                assert len(getattr(self, attr)) == len(shapes), (
                    f"The number of `{attr}` values does not match the "
                    "number shapes to cluster."
                )

        # Instantiate the PPN post-processor, if needed
        self.use_label_break_points = use_label_break_points
        assert not np.isscalar(
            break_shapes
        ), "Semantic classes to break should be provided as a list."
        if len(break_shapes) and not use_label_break_points:
            assert ppn_predictor is not None, (
                "If shapes are to be broken up using PPN points, "
                "must provide a PPN predictor configuration."
            )
            self.ppn_predictor = PPNPredictor(**ppn_predictor)

        # Initialize one clustering algorithm per class
        self.clusterers = []
        for k, c in enumerate(shapes):
            if c not in break_shapes:
                dbscan = spine_dbscan(
                    eps=self.eps[k],
                    min_samples=self.min_samples[k],
                    metric=self.metric[k],
                )
                clusterer = lambda x, _: dbscan.fit_predict(x)
            else:
                method = break_track_method
                if c != TRACK_SHP:
                    method = "masked_dbscan"
                clusterer = PointBreakClusterer(
                    eps=self.eps[k],
                    min_samples=self.min_samples[k],
                    metric=self.metric[k],
                    method=method,
                    mask_radius=self.break_mask_radius[k],
                )

            self.clusterers.append(clusterer)

    def forward(self, data, seg_pred, coord_label=None, **ppn_result):
        """Pass a batch of data through DBSCAN to form space clusters.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is 1 (charge/energy) if the clusters (`clusts`) are provided,
              or it needs to contain cluster labels to build them on the fly
        seg_pred : TensorBatch
            (N) Segmentation value for each data point
        coord_label : TensorBatch, optional
            Location of the true particle points
        **ppn_result : dict, optional
            Dictionary of outputs from the PPN model
        """
        # If some shapes must be broken up at their points of interest,
        # fetch them from the relevant location.
        points, point_shapes = None, None
        if len(self.break_shapes):
            if self.use_label_break_points:
                assert coord_label is not None, (
                    "If label points are to be used to break instance, "
                    "must provide them."
                )
                points = torch.cat(
                    (
                        coord_label.tensor[:, COORD_START_COLS],
                        coord_label.tensor[:, COORD_END_COLS],
                    ),
                    dim=1,
                ).reshape(-1, 3)
                point_shapes = torch.repeat_interleave(
                    coord_label.tensor[:, SHAPE_COL], 2
                )
                points = TensorBatch(points, 2 * coord_label.counts)
                point_shapes = TensorBatch(point_shapes, 2 * coord_label.counts)
            else:
                ppn_points = self.ppn_predictor(**ppn_result)
                points = TensorBatch(
                    ppn_points.tensor[:, COORD_COLS], ppn_points.counts
                )
                point_shapes = TensorBatch(
                    ppn_points.tensor[:, PPN_SHAPE_COL], ppn_points.counts
                )

        # Bring everything to numpy (DBSCAN cannot run on tensors)
        data_np = data.to_numpy()
        seg_pred_np = seg_pred.to_numpy()
        if points is not None:
            points_np = points.to_numpy()
            point_shapes_np = point_shapes.to_numpy()

        # Loop over the entries in the batch
        offsets = data.edges[:-1]
        clusts, shapes, counts, single_counts = [], [], [], []
        for b in range(data.batch_size):
            # Fetch the necessary data products, in numpy format
            voxels_b = data_np[b][:, COORD_COLS]
            seg_pred_b = seg_pred_np[b]
            if points is not None:
                points_b = points_np[b]
                point_shapes_b = point_shapes_np[b]

                # Exclude delta points, they do not help with clustering
                points_b = points_b[point_shapes_b != DELTA_SHP]

            # Loop over the shapes to cluster
            clusts_b, counts_b, shapes_b = [], [], []
            for k, s in enumerate(self.shapes):
                # Restrict the voxels to the current class
                break_class = s in self.break_shapes
                shape_mask = seg_pred_b == s
                if s == TRACK_SHP and break_class and self.track_include_delta:
                    shape_mask |= seg_pred_b == DELTA_SHP

                shape_index = np.where(shape_mask)[0]
                if not len(shape_index):
                    continue

                # Run clustering
                voxels_b_s = voxels_b[shape_index]
                labels = self.clusterers[k](voxels_b_s, points_b)

                # If delta points were added to track points, remove them
                if s == TRACK_SHP and break_class and self.track_include_delta:
                    labels[seg_pred_b[shape_index] == DELTA_SHP] = -1

                # Build clusters for this class
                clusts_b_s = []
                for c in np.unique(labels):
                    clust = np.where(labels == c)[0]
                    if c > -1 and len(clust) > self.min_size[k]:
                        clusts_b_s.append(int(offsets[b]) + clust)
                        counts_b.append(len(clust))

                clusts_b.extend(clusts_b_s)
                shapes_b.append(s * np.ones(len(clusts_b_s), dtype=np.int64))

            # Update the output
            clusts.extend(clusts_b)
            shapes.extend(shapes_b)
            counts.append(len(clusts_b))
            single_counts.extend(counts_b)

        # Initialize an IndexBatch and return it
        clusts_nb = np.empty(len(clusts), dtype=object)
        clusts_nb[:] = clusts

        index = IndexBatch(clusts_nb, offsets, counts, single_counts)
        if len(shapes):
            shapes = TensorBatch(np.concatenate(shapes), counts)
        else:
            shapes = TensorBatch(np.empty(0, dtype=np.int64), counts)

        return index, shapes
