"""Calorimetric interaction direction reconstruction module."""

import numpy as np

from spine.post.base import PostBase

__all__ = ["CalorimetricDirectionProcessor"]


class CalorimetricDirectionProcessor(PostBase):
    """Reconstructs the direction of each interaction.

    For each interaction, this algorithm takes the reconstructed vertex,
    projects all voxel positions onto a unit sphere centered at the vertex,
    and computes the charge-weighted sum of the resulting unit vectors.
    The final direction is the normalized sum.
    """

    name = "calorimetric_direction"

    aliases = ("reconstruct_calorimetric_direction", "reconstruct_nu_direction")

    _upstream = ("vertex",)

    def __init__(
        self,
        run_mode="both",
        truth_point_mode="points",
    ):
        """Initialize the calorimetric direction reconstruction parameters.

        Parameters
        ----------
        run_mode : str, optional
            If specified, tells whether the post-processor must run on
            reconstructed ('reco'), true ('true') or both objects
            ('both' or 'all')
        truth_point_mode : str, default 'points'
            If specified, tells which attribute of the truth interaction
            object to use to fetch its point coordinates
        """
        super().__init__("interaction", run_mode, truth_point_mode)

        if run_mode != "truth":
            self.update_keys({"points": True, "depositions": True})
        if run_mode != "reco":
            self.update_keys({self.truth_point_key: True})

    def process(self, data):
        """Reconstruct the direction for each interaction.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for k in self.interaction_keys:
            points_key = "points" if not "truth" in k else self.truth_point_key
            for inter in data[k]:
                self._reconstruct_direction(inter, data, points_key)

    def _reconstruct_direction(self, inter, data, points_key):
        """Reconstruct the direction for one interaction.

        Parameters
        ----------
        inter : RecoInteraction or TruthInteraction
            Interaction object
        data : dict
            Dictionary of data products
        points_key : str
            Key to access the point coordinates tensor
        """
        vertex = inter.vertex if not inter.is_truth else inter.reco_vertex
        if vertex is None or len(inter.particles) == 0:
            return

        all_indices = np.concatenate(
            [part.index for part in inter.particles if part.size > 0]
        )
        if len(all_indices) == 0:
            return

        points = data[points_key][all_indices]
        deps = data["depositions"][all_indices]

        vecs = points - vertex
        norms = np.linalg.norm(vecs, axis=1)
        mask = norms > 0
        if not np.any(mask):
            return

        dirs = vecs[mask] / norms[mask, np.newaxis]
        weights = deps[mask]
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return

        nu_dir = np.sum(dirs * weights[:, np.newaxis], axis=0) / total_weight
        norm = np.linalg.norm(nu_dir)
        if norm > 0:
            inter.nu_dir = nu_dir / norm
