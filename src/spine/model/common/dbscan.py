"""Module which does connected-components (dense) clustering using DBSCAN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
import torch

from spine.constants import (
    DELTA_SHP,
    MICHL_SHP,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.data import IndexBatch, TensorBatch
from spine.math.cluster import DBSCAN as spine_dbscan
from spine.utils.point_break_clustering import PointBreakClusterer
from spine.utils.ppn import PPNPredictor

T = TypeVar("T")
PPNResult: TypeAlias = TensorBatch | IndexBatch | Sequence[TensorBatch]


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
        eps: float | Sequence[float] = 1.8,
        min_samples: int | Sequence[int] = 1,
        min_size: int | Sequence[int] = 3,
        metric: str | Sequence[str] = "euclidean",
        shapes: Sequence[int] = (SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP),
        break_shapes: Sequence[int] = (TRACK_SHP,),
        break_mask_radius: float | Sequence[float] = 5.0,
        break_track_method: str = "masked_dbscan",
        use_label_break_points: bool = False,
        track_include_delta: bool = False,
        ppn_predictor: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the DBSCAN clustering algorithm.

        Parameters
        ----------
        eps : float or sequence of float, default 1.8
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
        min_samples : int or sequence of int, default 1
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point.
        min_size : int or sequence of int, default 3
            Minimum cluster size to stored in the final list of DBSCAN clusters
        metric : str or sequence of str, default "euclidean"
            Metric used to compute the pair-wise distances between space points
        shapes : sequence of int, default (0, 1, 2, 3)
            List of semantic classes to run DBSCAN on
        break_shapes : sequence of int, default (1,)
            List of semantic shapes for which to use PPN to break down
        break_mask_radius : float or sequence of float, default 5.0
            If using particle points to break up instances further, specifies
            the radius around each particle point which gets masked
        break_track_method : str, default 'masked_dbscan'
            If using particle points to break up tracks, specifies the method
        use_label_break_points : bool, default False
            Whether to use label points to break instances
        track_include_delta : bool, default False
            If `True`, include delta points along with track point when
            running DBSCAN on track points (limits artificial track breaks)
        ppn_predictor : dict, optional
            PPN post-processing configuration

        Raises
        ------
        ValueError
            If shape parameters are malformed, per-shape parameters have
            inconsistent lengths, or point breaking lacks a PPN configuration.
        """
        # Initialize the parent class
        super().__init__()

        # Store the DBSCAN clustering parameters
        if not isinstance(shapes, Sequence) or isinstance(shapes, (str, bytes)):
            raise ValueError("Semantic classes should be provided as a sequence.")
        if not isinstance(break_shapes, Sequence) or isinstance(
            break_shapes, (str, bytes)
        ):
            raise ValueError(
                "Semantic classes to break should be provided as a sequence."
            )
        self.shapes = list(shapes)
        self.break_shapes = list(break_shapes)
        self.eps = self._expand_parameter(eps, len(self.shapes), "eps")
        self.min_samples = self._expand_parameter(
            min_samples, len(self.shapes), "min_samples"
        )
        self.min_size = self._expand_parameter(min_size, len(self.shapes), "min_size")
        self.metric = self._expand_parameter(metric, len(self.shapes), "metric")
        self.break_mask_radius = self._expand_parameter(
            break_mask_radius, len(self.shapes), "break_mask_radius"
        )
        self.break_track_method = break_track_method
        self.track_include_delta = track_include_delta

        # Instantiate the PPN post-processor, if needed
        self.use_label_break_points = use_label_break_points
        self.ppn_predictor = None
        if len(self.break_shapes) > 0 and not use_label_break_points:
            if ppn_predictor is None:
                raise ValueError(
                    "If shapes are to be broken up using PPN points, "
                    "must provide a PPN predictor configuration."
                )
            self.ppn_predictor = PPNPredictor(**ppn_predictor)

        # Initialize one clustering algorithm per class
        self.clusterers = []
        for k, c in enumerate(self.shapes):
            if c not in self.break_shapes:
                dbscan = spine_dbscan(
                    eps=self.eps[k],
                    min_samples=self.min_samples[k],
                    metric=self.metric[k],
                )

                def _clusterer(x, _, algorithm=dbscan):
                    """Apply plain DBSCAN through the point-aware interface."""
                    return algorithm.fit_predict(x)

                clusterer = _clusterer

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

    @staticmethod
    def _expand_parameter(
        value: T | Sequence[T],
        size: int,
        name: str,
    ) -> list[T]:
        """Normalize a scalar or per-shape clustering parameter.

        Parameters
        ----------
        value : object or sequence
            Scalar value shared by every shape or one value per shape.
        size : int
            Number of semantic shapes.
        name : str
            Parameter name used in validation errors.

        Returns
        -------
        list
            One parameter value per semantic shape.

        Raises
        ------
        ValueError
            If a sequence does not contain exactly ``size`` values.
        """
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return [cast(T, value)] * size
        values = list(value)
        if len(values) != size:
            raise ValueError(
                f"The number of `{name}` values does not match the number "
                "of shapes to cluster."
            )
        return values

    def forward(
        self,
        data: TensorBatch,
        seg_pred: TensorBatch,
        coord_label: TensorBatch | None = None,
        **ppn_result: PPNResult,
    ) -> tuple[IndexBatch, TensorBatch]:
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

        Returns
        -------
        IndexBatch
            Batched voxel indices for every reconstructed fragment.
        TensorBatch
            Semantic shape assigned to each fragment.

        Raises
        ------
        TypeError
            If the PPN predictor does not return a batched tensor.
        ValueError
            If requested point labels or PPN outputs are unavailable.
        """
        # If some shapes must be broken up at their points of interest,
        # fetch them from the relevant location.
        points, point_shapes = None, None
        if len(self.break_shapes) > 0:
            if self.use_label_break_points:
                if coord_label is None:
                    raise ValueError(
                        "If label points are to be used to break instance, "
                        "must provide them."
                    )
                points_tensor = torch.cat(
                    (
                        coord_label.coordinates("start").torch_tensor(),
                        coord_label.coordinates("end").torch_tensor(),
                    ),
                    dim=1,
                ).reshape(-1, 3)
                point_shapes_tensor = torch.repeat_interleave(
                    coord_label.feature("shape").values.torch_tensor(), 2
                )
                points = TensorBatch(points_tensor, 2 * coord_label.counts)
                point_shapes = TensorBatch(point_shapes_tensor, 2 * coord_label.counts)
            else:
                if self.ppn_predictor is None:  # pragma: no cover
                    raise ValueError("PPN point breaking is not configured.")
                ppn_points = cast(TensorBatch, self.ppn_predictor(**ppn_result))
                points = ppn_points.coords
                point_shapes = ppn_points.feature("shape").values

        # Bring everything to numpy (DBSCAN cannot run on tensors)
        data_np = data.to_numpy()
        seg_pred_np = seg_pred.to_numpy()
        points_np = None
        point_shapes_np = None
        if points is not None and point_shapes is not None:
            points_np = points.to_numpy()
            point_shapes_np = point_shapes.to_numpy()

        # Loop over the entries in the batch
        offsets = data_np.edges[:-1]
        clusts, shapes, counts, single_counts = [], [], [], []
        for b in range(data.batch_size):
            # Fetch the necessary data products, in numpy format
            voxels_b = data_np.coords[b]
            seg_pred_b = seg_pred_np[b]
            points_b = np.empty((0, voxels_b.shape[1]), dtype=voxels_b.dtype)
            if points_np is not None and point_shapes_np is not None:
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
                if len(shape_index) == 0:
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
                    if c > -1 and len(clust) >= self.min_size[k]:
                        clusts_b_s.append(int(offsets[b]) + shape_index[clust])
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

        index = IndexBatch(clusts_nb, data_np.counts, counts, single_counts)
        if len(shapes) > 0:
            shapes = TensorBatch(np.concatenate(shapes), counts)
        else:
            shapes = TensorBatch(np.empty(0, dtype=np.int64), counts)

        return index, shapes
