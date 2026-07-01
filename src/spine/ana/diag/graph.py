"""Diagnostic analysis tools for graph clustering edge lengths.

This script evaluates the expected length between fragments/particles that
belong to the same group (particle or interaction groups) in order to optimize
the edge length.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spine.ana.base import AnaBase
from spine.utils.gnn.network import inter_cluster_distance

__all__ = ["GraphEdgeLengthAna"]


class GraphEdgeLengthAna(AnaBase):
    """Measure distances between constituents of particle/interaction groups.

    The output can be used to tune graph-edge construction thresholds for
    fragment-to-particle and particle-to-interaction clustering.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "graph_edge_length"

    def __init__(
        self,
        time_window: Sequence[float] | None = None,
        obj_type: str | Sequence[str] | None = ("particle", "interaction"),
        run_mode: str = "truth",
        truth_point_mode: str = "points",
        **kwargs: Any,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        time_window : Sequence[float], optional
            True-object time window to include in the diagnostic.
        obj_type : str or Sequence[str], default ('particle', 'interaction')
            Object aggregation levels to evaluate.
        run_mode : str, default 'truth'
            Whether to run on reconstructed, truth, or both object collections.
        truth_point_mode : str, default 'points'
            Truth point coordinate attribute to use when evaluating truth objects.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`.
        """
        obj_types: list[str]
        if isinstance(obj_type, str):
            obj_types = [obj_type]
        elif obj_type is None:
            obj_types = []
        else:
            obj_types = list(obj_type)

        # If particle clustering edges are to be evaluated, must provide fragments
        if "particle" in obj_types and "fragment" not in obj_types:
            obj_types.append("fragment")

        # If interaction clustering edges are to be evaluated, must provide particles
        if "interaction" in obj_types and "particle" not in obj_types:
            obj_types.append("particle")

        # Initialize the parent class
        super().__init__(
            obj_type=obj_types,
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            **kwargs,
        )

        # Store the parameters
        normalized_time_window: tuple[float, float] | None = None
        if time_window is not None:
            if not isinstance(time_window, Sequence) or len(time_window) != 2:
                raise ValueError(
                    "Time window must be specified as an array of two values."
                )
            if run_mode != "truth":
                raise ValueError("Cannot restrict timing of reconstructed particles.")
            normalized_time_window = (time_window[0], time_window[1])
        self.time_window = normalized_time_window

        # Add necessary point matrices
        if run_mode != "truth":
            self.update_keys({"points": True})
        if run_mode != "reco":
            self.update_keys({self.truth_point_key: True})

        # Initialize the CSV writer(s) you want
        for key in self.particle_keys + self.interaction_keys:
            self.initialize_writer(key)

    def process(self, data: Mapping[str, Any]) -> None:
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over aggregated objects (reco/truth particles/interactions)
        for key in self.particle_keys + self.interaction_keys:
            # Loop over all objects of that type
            for obj in data[key]:
                # Extract the list of constituents, feed to distance extractor
                points = (
                    data["points"] if not obj.is_truth else data[self.truth_point_key]
                )
                if hasattr(obj, "fragments"):
                    self.store_distances(key, points, obj, obj.fragments)
                else:
                    self.store_distances(key, points, obj, obj.particles)

    def store_distances(
        self,
        key: str,
        points: NDArray[np.floating],
        group: Any,
        constituents: Sequence[Any],
    ) -> None:
        """Store pairwise distances between constituents of one aggregate object.

        Parameters
        ----------
        key : str
            Name of the aggregated object
        points : np.ndarray
            Set of point locations for this object collection
        group : object
            Aggregated object (particle or interaction)
        constituents : Sequence[object]
            Constituents that make up the aggregated object
        """
        # Skip if there is one or less constituent
        if len(constituents) < 2:
            return

        # If there is a time window, throw out true out-of-time objects
        if self.time_window is not None and group.is_truth:
            if group.t < self.time_window[0] or group.t > self.time_window[1]:
                return

        # Compute the pair-wise distances constituent-to-constituent
        dists = np.asarray(
            inter_cluster_distance(points, [const.index for const in constituents]),
            dtype=np.float64,
        )
        dists[np.diag_indices_from(dists)] = 1e9

        # Compute the minimum distance within each shape pair
        shapes = np.asarray([const.shape for const in constituents], dtype=np.int64)
        unique_shapes = np.unique(shapes)
        shape_indexes: list[NDArray[np.int64]] = [
            np.where(shapes == s)[0] for s in unique_shapes
        ]
        for i, shape_i in enumerate(unique_shapes):
            index_i = shape_indexes[i]
            for offset, shape_j in enumerate(unique_shapes[i:]):
                # Skip single shape pairs if there is only one
                index_j = shape_indexes[i + offset]
                if shape_i == shape_j and len(index_i) < 2:
                    continue

                # Loop over constituents, store info
                dists_shape = dists[index_i]
                argmins = np.argmin(dists_shape[:, index_j], axis=1)
                for k, argmin in enumerate(argmins):
                    sink_index = int(index_j[argmin])
                    const_i, const_j = (
                        constituents[index_i[k]],
                        constituents[sink_index],
                    )
                    out = {
                        "id": group.id,
                        "source_id": const_i.id,
                        "sink_id": const_j.id,
                        "source_shape": const_i.shape,
                        "sink_shape": const_j.shape,
                        "length": dists_shape[k, sink_index],
                    }
                    self.append(key, **out)
