"""Turn SPICE spatial embeddings into particle-fragment candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import torch

from spine.constants.factory import enum_factory
from spine.data import IndexBatch, TensorBatch

__all__ = ["SPICEClusterer"]


class SPICEClusterer:
    """Produce clusters by iteratively growing Gaussian embedding masks.

    The highest-seediness unassigned voxel defines a candidate centroid and
    width. Voxels of the same semantic shape whose Gaussian membership exceeds
    the configured probability are assigned to that cluster. This is the
    inference counterpart of :class:`spine.model.spice.SPICELoss`.
    """

    def __init__(
        self,
        shapes: Sequence[int | str],
        seed_threshold: Real | Sequence[Real] | Mapping[int | str, Real] = 0.0,
        probability_threshold: Real | Sequence[Real] | Mapping[int | str, Real] = 0.5,
        min_size: int = 2,
        assign_all: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """Initialize embedding-mask clustering.

        Parameters
        ----------
        shapes : sequence of int or str
            Semantic shapes owned by SPICE.
        seed_threshold, probability_threshold : scalar, sequence or mapping
            Per-shape thresholds. Sequences follow the order of ``shapes``;
            mappings may use integer or named shape keys.
        min_size : int, default 2
            Minimum number of voxels in a retained cluster.
        assign_all : bool, default True
            Assign voxels outside every accepted mask to their most probable
            accepted cluster of the same shape.
        eps : float, default 1e-6
            Minimum positive Gaussian width.
        """
        self.shapes = tuple(self._shape(shape) for shape in shapes)
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("SPICE clustering shapes must be unique.")
        if min_size < 1:
            raise ValueError("`min_size` must be positive.")
        if eps <= 0.0:
            raise ValueError("`eps` must be positive.")

        self.seed_thresholds = self._thresholds(seed_threshold, "seed_threshold")
        self.probability_thresholds = self._thresholds(
            probability_threshold,
            "probability_threshold",
        )
        if any(not 0.0 <= value <= 1.0 for value in self.seed_thresholds.values()):
            raise ValueError("Seed thresholds must lie in [0, 1].")
        if any(
            not 0.0 <= value <= 1.0 for value in self.probability_thresholds.values()
        ):
            raise ValueError("Probability thresholds must lie in [0, 1].")

        self.min_size = min_size
        self.assign_all = assign_all
        self.eps = eps

    @staticmethod
    def _shape(shape: int | str) -> int:
        """Resolve a semantic shape identifier to its integer value."""
        return enum_factory("shape", shape) if isinstance(shape, str) else int(shape)

    def _thresholds(
        self,
        values: Real | Sequence[Real] | Mapping[int | str, Real],
        name: str,
    ) -> dict[int, float]:
        """Normalize scalar, ordered, or keyed per-shape thresholds."""
        if isinstance(values, Real):
            return {shape: float(values) for shape in self.shapes}
        if isinstance(values, Mapping):
            parsed = {self._shape(key): float(value) for key, value in values.items()}
            missing = set(self.shapes) - set(parsed)
            extra = set(parsed) - set(self.shapes)
            if missing or extra:
                raise ValueError(
                    f"`{name}` keys must exactly match SPICE shapes; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}."
                )
            return parsed

        values = tuple(values)
        if len(values) != len(self.shapes):
            raise ValueError(f"`{name}` must provide one value per SPICE shape.")
        return {shape: float(value) for shape, value in zip(self.shapes, values)}

    def _cluster_shape(
        self,
        embeddings: torch.Tensor,
        margins: torch.Tensor,
        seediness: torch.Tensor,
        indexes: torch.Tensor,
        shape: int,
    ) -> list[torch.Tensor]:
        """Cluster one event/shape slice and return filtered-tensor indexes."""
        available = torch.ones(len(indexes), dtype=torch.bool, device=indexes.device)
        clusters: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []

        while bool(torch.any(available)):
            scores = seediness.clone()
            scores[~available] = -1.0
            seed = int(torch.argmax(scores))
            if float(scores[seed]) < self.seed_thresholds[shape]:
                break

            width = torch.clamp(margins[seed], min=self.eps)
            squared_distance = torch.sum((embeddings - embeddings[seed]) ** 2, dim=1)
            probability = torch.exp(-squared_distance / (2.0 * width**2))
            members = available & (probability >= self.probability_thresholds[shape])

            # Reject undersized masks, but retire the seed so another candidate
            # can still recover a valid neighboring fragment.
            if int(torch.count_nonzero(members)) < self.min_size:
                available[seed] = False
                continue

            clusters.append(indexes[members])
            probabilities.append(probability)
            available[members] = False

        if self.assign_all and len(clusters) > 0 and bool(torch.any(available)):
            assignments = torch.stack(probabilities, dim=1).argmax(dim=1)
            for cluster_id, cluster in enumerate(clusters):
                remainder = indexes[available & (assignments == cluster_id)]
                if len(remainder) > 0:
                    clusters[cluster_id] = torch.cat((cluster, remainder))

        return clusters

    def __call__(
        self,
        embeddings: TensorBatch,
        margins: TensorBatch,
        seediness: TensorBatch,
        seg_label: TensorBatch,
    ) -> tuple[IndexBatch, TensorBatch]:
        """Cluster a shape-filtered SPICE output batch.

        All inputs must share the filtered voxel row domain. Returned cluster
        indexes refer to that same domain and can therefore be restored with
        the model's ``filter_index`` output.
        """
        counts = embeddings.counts
        if not (
            torch.equal(counts, margins.counts)
            and torch.equal(counts, seediness.counts)
            and torch.equal(counts, seg_label.counts)
        ):
            raise ValueError("All SPICE clustering inputs must share batch counts.")

        embedding_values = embeddings.torch_tensor()
        margin_values = margins.torch_tensor().flatten()
        seed_values = seediness.torch_tensor().flatten()
        shape_values = seg_label.torch_tensor().flatten().long()
        clusters: list[torch.Tensor] = []
        cluster_shapes: list[int] = []
        cluster_counts: list[int] = []

        for batch_id in range(embeddings.batch_size):
            lower = int(embeddings.edges[batch_id])
            upper = int(embeddings.edges[batch_id + 1])
            event_count = 0
            for shape in self.shapes:
                local = torch.where(shape_values[lower:upper] == shape)[0] + lower
                if len(local) < self.min_size:
                    continue
                shape_clusters = self._cluster_shape(
                    embedding_values[local],
                    margin_values[local],
                    seed_values[local],
                    local,
                    shape,
                )
                clusters.extend(shape_clusters)
                cluster_shapes.extend([shape] * len(shape_clusters))
                event_count += len(shape_clusters)
            cluster_counts.append(event_count)

        device = embedding_values.device
        counts_tensor = torch.tensor(cluster_counts, dtype=torch.long, device=device)
        sizes = torch.tensor(
            [len(cluster) for cluster in clusters],
            dtype=torch.long,
            device=device,
        )
        empty_index = torch.empty(0, dtype=torch.long, device=device)
        clusts = IndexBatch(
            clusters,
            spans=counts,
            counts=counts_tensor,
            single_counts=sizes,
            default=empty_index,
        )
        shapes = TensorBatch(
            torch.tensor(cluster_shapes, dtype=torch.long, device=device),
            counts_tensor,
        )
        return clusts, shapes
