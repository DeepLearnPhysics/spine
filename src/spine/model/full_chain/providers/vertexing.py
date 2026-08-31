"""Interaction-level vertex prediction provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

import spine.math as sm
from spine.data import IndexBatch, Meta, TensorBatch, TensorSchema
from spine.model.grappa.vertex import decode_vertex_positions

from ..point import PointBatch
from ..registry import ProviderSpec, register_provider
from ..stage import ChainStage
from ..state import ChainState, StageResult

__all__ = ["InteractionVertexingStage"]


class InteractionVertexingStage(ChainStage):
    """Reduce raw point or particle proposals to one vertex per interaction.

    PPN mode pools voxel-level proposals inside each reconstructed interaction.
    GrapPA mode selects the particle with the largest predicted primary
    probability in each interaction and uses its decoded vertex regression.
    Both modes publish products aligned one-to-one with
    ``interaction_clusts``.
    """

    provides = frozenset({"interaction_vertices", "interaction_vertex_scores"})

    def __init__(
        self,
        name: str,
        mode: str,
        score_threshold: float = 0.5,
        pool_radius: float = 1.999,
        pool_score_fn: str = "max",
        normalize_positions: bool = False,
        use_anchor_points: bool = False,
    ) -> None:
        """Initialize an interaction vertex reducer.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"ppn", "grappa"}
            Source of raw vertex predictions.
        score_threshold : float, default 0.5
            Minimum PPN foreground probability used to form candidates. If no
            proposal passes, the most probable interaction voxel is retained.
        pool_radius : float, default 1.999
            Maximum separation between PPN positions in one candidate cluster.
        pool_score_fn : {"max", "mean"}, default "max"
            Function used to rank pooled PPN candidate clusters.
        normalize_positions : bool, default False
            Decode GrapPA regression values in image-normalized coordinates.
        use_anchor_points : bool, default False
            Decode GrapPA regression values relative to particle endpoints.

        Raises
        ------
        ValueError
            If the mode or a numerical/selection option is invalid.
        """
        super().__init__(name)
        if mode not in {"ppn", "grappa"}:
            raise ValueError("Interaction vertexing mode must be `ppn` or `grappa`.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("`score_threshold` must be between zero and one.")
        if pool_radius <= 0.0:
            raise ValueError("`pool_radius` must be positive.")
        if pool_score_fn not in {"max", "mean"}:
            raise ValueError("`pool_score_fn` must be `max` or `mean`.")
        if mode == "ppn" and (normalize_positions or use_anchor_points):
            raise ValueError(
                "GrapPA position decoding options cannot be used in PPN mode."
            )

        self.mode = mode
        self.score_threshold = score_threshold
        self.pool_radius = pool_radius
        self.pool_score_fn = pool_score_fn
        self.normalize_positions = normalize_positions
        self.use_anchor_points = use_anchor_points

        common = {"interaction_clusts"}
        if mode == "ppn":
            self.requires = frozenset(common | {"point_data", "vertex_proposals"})
            self.optional = frozenset()
        else:
            required = common | {
                "particle_vertex_proposals",
                "particle_interaction_ids",
            }
            if normalize_positions:
                required.add("meta")
            if use_anchor_points:
                required.update(
                    {"particle_vertex_start_points", "particle_vertex_end_points"}
                )
            self.requires = frozenset(required)
            self.optional = frozenset()

    @staticmethod
    def _make_result(
        vertices: list[torch.Tensor],
        scores: list[torch.Tensor],
        interactions: IndexBatch,
        reference: torch.Tensor,
    ) -> StageResult:
        """Package interaction-aligned vertices and confidence scores."""
        if vertices:
            vertex_tensor = torch.stack(vertices)
            score_tensor = torch.stack(scores)
        else:
            vertex_tensor = reference.new_empty((0, 3))
            score_tensor = reference.new_empty((0,))

        vertex_batch = TensorBatch(
            vertex_tensor,
            interactions.counts,
            coord_cols=(0, 1, 2),
            schema=TensorSchema(coordinate_groups={"vertex": (0, 1, 2)}),
        )
        score_batch = TensorBatch(
            score_tensor,
            interactions.counts,
            schema=TensorSchema(
                feature_fields={"vertex_score": (0,)},
                feats_only=True,
            ),
        )
        products = {
            "interaction_vertices": vertex_batch,
            "interaction_vertex_scores": score_batch,
        }
        return StageResult(products, dict(products))

    def _pool_ppn_candidates(
        self,
        positions: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and pool the most probable PPN vertex candidate."""
        selected = torch.where(scores >= self.score_threshold)[0]
        if len(selected) == 0:
            selected = torch.argmax(scores).reshape(1)

        selected_positions = positions[selected]
        selected_scores = scores[selected]
        labels = sm.cluster.dbscan(
            selected_positions.detach().cpu().numpy(),
            eps=self.pool_radius,
            min_samples=1,
        )

        clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        pooled_scores = []
        for cluster in clusters:
            index = torch.as_tensor(
                cluster,
                dtype=torch.long,
                device=scores.device,
            )
            cluster_scores = selected_scores[index]
            pooled_scores.append(
                torch.amax(cluster_scores)
                if self.pool_score_fn == "max"
                else torch.mean(cluster_scores)
            )

        cluster_id = int(torch.argmax(torch.stack(pooled_scores)).item())
        cluster = torch.as_tensor(
            clusters[cluster_id],
            dtype=torch.long,
            device=scores.device,
        )
        cluster_scores = selected_scores[cluster]
        weights = cluster_scores / torch.sum(cluster_scores)
        vertex = torch.sum(selected_positions[cluster] * weights[:, None], dim=0)
        return vertex, pooled_scores[cluster_id]

    def _from_ppn(
        self,
        point_data: PointBatch,
        proposals: TensorBatch,
        interactions: IndexBatch,
    ) -> StageResult:
        """Reduce row-aligned voxel proposals inside each interaction."""
        data = point_data.data
        if proposals.shape[0] != data.shape[0]:
            raise ValueError(
                "PPN vertex proposals must be row-aligned with `point_data`."
            )
        if proposals.batch_size != interactions.batch_size:
            raise ValueError(
                "PPN proposals and interactions must have the same batch size."
            )

        offsets = proposals.feature("offsets")
        logits = proposals.feature("vertex_logits")
        vertices, vertex_scores = [], []
        for batch_id in range(interactions.batch_size):
            coordinates = data.coords[batch_id]
            offsets_b = offsets[batch_id]
            scores_b = torch.softmax(logits[batch_id], dim=1)[:, 1]
            for interaction in interactions[batch_id]:
                if len(interaction) == 0:
                    raise ValueError(
                        "Cannot predict a vertex for an empty interaction."
                    )
                index = torch.as_tensor(
                    interaction,
                    dtype=torch.long,
                    device=data.device,
                )
                positions = coordinates[index] + 0.5 + offsets_b[index]
                vertex, score = self._pool_ppn_candidates(
                    positions,
                    scores_b[index],
                )
                vertices.append(vertex)
                vertex_scores.append(score)
        return self._make_result(
            vertices,
            vertex_scores,
            interactions,
            proposals.torch_tensor(),
        )

    def _from_grappa(
        self,
        proposals: TensorBatch,
        interaction_ids: TensorBatch,
        interactions: IndexBatch,
        meta: Sequence[Meta] | None,
        start_points: TensorBatch | None,
        end_points: TensorBatch | None,
    ) -> StageResult:
        """Select the highest-primary-probability particle per interaction."""
        if proposals.shape[1] != 5:
            raise ValueError(
                "GrapPA vertex proposals must contain two primary logits and "
                "three regression values."
            )
        if proposals.shape[0] != interaction_ids.shape[0]:
            raise ValueError(
                "GrapPA vertex proposals and interaction IDs must be aligned."
            )
        primary_logits, regression = torch.tensor_split(
            proposals.torch_tensor(),
            [2],
            dim=1,
        )
        regression_batch = TensorBatch(regression, proposals.counts)
        decoded = decode_vertex_positions(
            regression_batch,
            start_points=start_points,
            end_points=end_points,
            meta=meta,
            normalize_positions=self.normalize_positions,
            use_anchor_points=self.use_anchor_points,
            restore_absolute=True,
        )

        vertices, vertex_scores = [], []
        for batch_id in range(interactions.batch_size):
            group_ids = interaction_ids[batch_id].flatten().long()
            group_values = torch.unique(group_ids, sorted=True)
            if len(group_values) != len(interactions[batch_id]):
                raise ValueError(
                    "GrapPA interaction IDs do not match `interaction_clusts`."
                )
            probabilities = torch.softmax(
                primary_logits[
                    proposals.edges[batch_id] : proposals.edges[batch_id + 1]
                ],
                dim=1,
            )[:, 1]
            positions = decoded[batch_id]
            for group_id in group_values:
                members = torch.where(group_ids == group_id)[0]
                local_id = members[torch.argmax(probabilities[members])]
                vertices.append(positions[local_id])
                vertex_scores.append(probabilities[local_id])
        return self._make_result(
            vertices,
            vertex_scores,
            interactions,
            proposals.torch_tensor(),
        )

    def forward(self, state: ChainState) -> StageResult:
        """Produce one interaction vertex from the configured proposal source."""
        interactions: IndexBatch = state.require("interaction_clusts", self.name)
        with torch.no_grad():
            if self.mode == "ppn":
                return self._from_ppn(
                    state.require("point_data", self.name),
                    state.require("vertex_proposals", self.name),
                    interactions,
                )
            return self._from_grappa(
                state.require("particle_vertex_proposals", self.name),
                state.require("particle_interaction_ids", self.name),
                interactions,
                state.get("meta"),
                state.get("particle_vertex_start_points"),
                state.get("particle_vertex_end_points"),
            )


def build_interaction_vertexing_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build a non-trainable interaction vertex reduction stage.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Vertex-reduction options forwarded to :class:`InteractionVertexingStage`.
    owner : object
        Full-chain owner, unused because the stage is non-trainable.
    """
    del owner
    return InteractionVertexingStage(name, **config)


PROVIDER_SPEC = register_provider(
    ProviderSpec("interaction_vertexing", build_interaction_vertexing_stage)
)
