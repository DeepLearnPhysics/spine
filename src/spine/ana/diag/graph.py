"""Module with methods to characterize

This script evaluates the expected length between fragments/particles that
belong to the same group (particle or interaction groups) in order to optimize
the edge length
"""

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.network import inter_cluster_distance

__all__ = ["GraphEdgeLengthAna"]


class GraphEdgeLengthAna(AnaBase):
    """Description of what the analysis script is supposed to be doing.

    This script evaluates the expected length between fragments/particles that
    belong to the same group (particle or interaction groups) in order to optimize
    the edge length
    """

    # Name of the analysis script (as specified in the configuration)
    name = "graph_edge_length"

    def __init__(
        self,
        time_window=None,
        obj_type=("particle", "interaction"),
        run_mode="truth",
        truth_point_mode="points",
        **kwargs,
    ):
        """Initialize the analysis script.

        Parameters
        ----------
        """
        # If particle clustering edges are to be evaluated, must provide fragments
        if obj_type == "particle":
            obj_type = ["fragment", "particle"]
        elif "particle" in obj_type and "fragment" not in obj_type:
            obj_type.append("fragment")

        # If interaction clustering edges are to be evaluated, must provide particles
        if obj_type == "interaction":
            obj_type = ["particle", "interaction"]
        elif "interaction" in obj_type and "particle" not in obj_type:
            obj_type.append("particle")

        # Initialize the parent class
        super().__init__(
            obj_type=obj_type,
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            **kwargs,
        )

        # Store the parameters
        self.time_window = time_window
        if self.time_window is not None and run_mode != "truth":
            raise ValueError("Cannot restrict timeing of reconstructed particles")

        # Add necessary point matrices
        if run_mode != "truth":
            self.update_keys({"points": True})
        if run_mode != "reco":
            self.update_keys({self.truth_point_key: True})

        # Initialize the CSV writer(s) you want
        for key in self.particle_keys + self.interaction_keys:
            self.initialize_writer(key)

    def process(self, data):
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over aggregarted objects (reco/truth particles/interactions)
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

    def store_distances(self, key, points, group, constituents):
        """Store the pair-wise distances between constituents of an aggregated object.

        Parameters
        ----------
        key : str
            Name of the aggregated object
        points : np.ndarray
            Set of point locations for this this of object
        group : object
            Aggregated object (particle or interaction)
        constituents : List[object]
            List of constiuents that make up the aggregated object (fragment or particle)
        """
        # Skip if there is one or less constituent
        if len(constituents) < 2:
            return

        # If there is a time window, throw out true out-of-time objects
        if self.time_window is not None and group.is_truth:
            if group.t < self.time_window[0] or group.t > self.time_window[1]:
                return

        # Compute the pair-wise distances constituent-to-constituent
        dists = inter_cluster_distance(points, [const.index for const in constituents])
        dists[np.diag_indices_from(dists)] = 1e9

        # Compute the minimum distance within each shape pair
        shapes = np.array([const.shape for const in constituents])
        unique_shapes = np.unique(shapes)
        shape_indexes = [np.where(shapes == s)[0] for s in unique_shapes]
        for i, shape_i in enumerate(unique_shapes):
            index_i = shape_indexes[i]
            for j, shape_j in enumerate(unique_shapes[i:]):
                # Skip single shape pairs if there is only one
                index_j = shape_indexes[j]
                if shape_i == shape_j and len(index_i) < 2:
                    continue

                # Loop over constituents, store info
                dists_shape = dists[index_i]
                argmins = np.argmin(dists_shape[:, index_j], axis=1)
                for k, argmin in enumerate(argmins):
                    const_i, const_j = (
                        constituents[index_i[k]],
                        constituents[index_j[argmin]],
                    )
                    out = {
                        "id": group.id,
                        "source_id": const_i.id,
                        "sink_id": const_j.id,
                        "source_shape": const_i.shape,
                        "sink_shape": const_j.shape,
                        "length": dists_shape[k, argmin],
                    }
                    self.append(key, **out)
