"""Module to evaluate diagnostic metrics on tracks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spine.ana.base import AnaBase
from spine.constants import MUON_PID, TRACK_SHP
from spine.math.distance import cdist

__all__ = ["TrackCompletenessAna"]


class TrackCompletenessAna(AnaBase):
    """Identify gaps in tracks and measure their cumulative missing length.

    This is a useful diagnostic tool to evaluate the space-point efficiency
    on tracks (a good standard candle, as tracks should have no gaps in
    a perfectly efficient detector).
    """

    # Name of the analysis script (as specified in the configuration)
    name = "track_completeness"

    def __init__(
        self,
        time_window: Sequence[float] | None = None,
        length_threshold: float | None = 10,
        include_pids: Sequence[int] | None = (MUON_PID,),
        run_mode: str = "both",
        truth_point_mode: str = "points",
        **kwargs,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        time_window : Sequence[float], optional
            Time window within which to include particle (only works for `truth`)
        length_threshold : float, optional
            Minimum length of tracks to consider, in cm
        include_pids : Sequence[int], optional
            Particle IDs to include in the analysis
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(
            obj_type="particle",
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            **kwargs,
        )

        # Store the time window
        normalized_time_window: tuple[float, float] | None = None
        if time_window is not None:
            if not isinstance(time_window, Sequence) or len(time_window) != 2:
                raise ValueError(
                    "Time window must be specified as an array of two values."
                )
            if run_mode != "truth":
                raise ValueError("Time of reconstructed particle is unknown.")
            normalized_time_window = (time_window[0], time_window[1])
        self.time_window = normalized_time_window

        # Store the length threshold and included PIDs
        self.length_threshold = length_threshold
        self.include_pids = include_pids

        # Make sure the metadata is provided (rasterization needed)
        self.update_keys({"meta": True})

        # Initialize the CSV writer(s) you want
        for prefix in self.prefixes:
            self.initialize_writer(prefix)

    def process(self, data: Mapping[str, Any]) -> None:
        """Evaluate track completeness for tracks in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the pixel size in this image (assume cubic cells)
        if not np.all(data["meta"].size[0] == data["meta"].size):
            raise ValueError("Non-cubic pixels not supported.")
        pixel_size = data["meta"].size[0]

        # Loop over the types of particle data products
        for key in self.obj_keys:
            # Fetch the prefix ('reco' or 'truth')
            prefix = key.split("_")[0]

            # Loop over particle objects
            for part in data[key]:
                # Check that the particle is a track
                if part.shape != TRACK_SHP:
                    continue

                # If needed, check on the particle time
                if self.time_window is not None:
                    if part.t < self.time_window[0] or part.t > self.time_window[1]:
                        continue

                # If needed, check on the particle PID
                if self.include_pids is not None:
                    if part.pid not in self.include_pids:
                        continue

                # Initialize the particle dictionary
                comp_dict = {"particle_id": part.id}

                # Fetch the particle point coordinates
                points = self.get_points(part)

                # Find start/end points, collapse onto track cluster
                start = points[np.argmin(cdist(part.start_point[None, :], points))]
                end = points[np.argmin(cdist(part.end_point[None, :], points))]

                # Add the direction of the track
                vec = end - start
                length = np.linalg.norm(vec)
                if length:
                    vec /= length

                # If needed, check on the particle length
                if self.length_threshold is not None:
                    if length < self.length_threshold:
                        continue

                comp_dict["size"] = len(points)
                comp_dict["length"] = length
                comp_dict.update({"dir_x": vec[0], "dir_y": vec[1], "dir_z": vec[2]})
                comp_dict.update(
                    {"start_x": start[0], "start_y": start[1], "start_z": start[2]}
                )
                comp_dict.update({"end_x": end[0], "end_y": end[1], "end_z": end[2]})

                # Chunk out the track along gaps, estimate gap length
                chunk_labels = self.cluster_track_chunks(points, start, end, pixel_size)
                gaps = self.sequential_cluster_distances(points, chunk_labels, start)

                # Substract minimum gap distance due to rasterization
                min_gap = pixel_size * np.sum(np.abs(vec))
                gaps -= min_gap

                # Store gap information
                comp_dict["num_gaps"] = len(gaps)
                comp_dict["gap_length"] = np.sum(gaps)
                comp_dict["gap_frac"] = np.sum(gaps) / length

                # Append the dictionary to the CSV log
                self.append(prefix, **comp_dict)

    @staticmethod
    def cluster_track_chunks(
        points: NDArray[np.floating],
        start_point: NDArray[np.floating],
        end_point: NDArray[np.floating],
        pixel_size: float,
    ) -> NDArray[np.integer]:
        """Split a track into chunks at gaps along its main axis.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) List of track cluster point coordinates
        start_point : np.ndarray
            (3) Start point of the track cluster
        end_point : np.ndarray
            (3) End point of the track cluster
        pixel_size : float
            Dimension of one pixel, used to identify what is big enough to
            constitute a break

        Returns
        -------
        np.ndarray
            (N) Track chunk labels
        """
        # Project and cluster on the projected axis
        direction = (end_point - start_point) / np.linalg.norm(end_point - start_point)
        min_gap = pixel_size * np.sum(np.abs(direction))
        projs = np.dot(points - start_point, direction)
        perm = np.argsort(projs)
        seps = projs[perm][1:] - projs[perm][:-1]
        breaks = np.where(seps > min_gap * 1.1)[0] + 1
        cluster_labels = np.empty(len(projs), dtype=np.int64)
        for i, index in enumerate(np.split(np.arange(len(projs)), breaks)):
            cluster_labels[perm[index]] = i

        return cluster_labels

    @staticmethod
    def sequential_cluster_distances(
        points: NDArray[np.floating],
        labels: NDArray[np.integer],
        start_point: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Order clusters in order of distance from a starting point, compute
        the distances between successive clusters.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) List of track cluster point coordinates
        labels : np.ndarray
            (N) Track chunk labels
        start_point : np.ndarray
            (3) Start point of the track cluster
        """
        # If there's only one cluster, nothing to do here
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return np.empty(0, dtype=np.float64)

        # Order clusters
        start_dist = cdist(start_point[None, :], points).flatten()
        start_clust_dist = np.empty(len(unique_labels), dtype=np.float64)
        for i, c in enumerate(unique_labels):
            start_clust_dist[i] = np.min(start_dist[labels == c])
        ordered_labels = unique_labels[np.argsort(start_clust_dist)]

        # Compute the intercluster distance and relative angle
        n_gaps = len(ordered_labels) - 1
        dists = np.empty(n_gaps, dtype=np.float64)
        for i in range(n_gaps):
            points_i = points[labels == ordered_labels[i]]
            points_j = points[labels == ordered_labels[i + 1]]
            dists[i] = np.min(cdist(points_i, points_j))

        return dists
