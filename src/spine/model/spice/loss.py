"""Spatial embedding objective for the SPICE clustering model."""

from __future__ import annotations

from typing import Any

import torch

from spine.config.factory import Config
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory

__all__ = ["SPICELoss"]


class SPICELoss(torch.nn.Module):
    """Supervise SPICE embeddings, margins, and seediness predictions.

    For each batch entry and semantic class, the loss constructs one Gaussian
    mask per target cluster. Cluster centroids and widths are estimated from
    the predicted embeddings and margins. Seediness is trained to reproduce
    each voxel's probability under its own target-cluster mask.
    """

    def __init__(
        self,
        spice: dict[str, Any],
        spice_loss: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SPICE objective.

        Parameters
        ----------
        spice : dict
            SPICE model configuration. Present for the shared model/loss
            constructor contract.
        spice_loss : dict, optional
            Loss configuration.
        """
        # Initialize the parent class
        super().__init__()

        # Process the model-dependent contract before the loss parameters
        self.process_model_config(**spice)
        self.process_loss_config(**({} if spice_loss is None else spice_loss))

    def process_model_config(
        self,
        margin_dim: int = 1,
        seediness_dim: int = 1,
        **_kwargs: Any,
    ) -> None:
        """Validate model properties required by the loss.

        Parameters
        ----------
        margin_dim : int, default 1
            Number of predicted cluster margins.
        seediness_dim : int, default 1
            Number of predicted seediness scores.
        **kwargs : dict, optional
            Other model parameters unused by the loss.
        """
        if margin_dim != 1 or seediness_dim != 1:
            raise ValueError(
                "SPICELoss requires one margin and one seediness value per voxel."
            )

    def process_loss_config(
        self,
        mask_loss: Config = "bce_logits",
        seed_loss: Config = "l1",
        embedding_weight: float = 1.0,
        seediness_weight: float = 1.0,
        smoothing_weight: float = 1.0,
        inter_weight: float = 1.0,
        inter_margin: float = 0.2,
        min_voxels: int = 2,
        eps: float = 1e-6,
    ) -> None:
        """Configure the individual SPICE loss terms.

        Parameters
        ----------
        mask_loss : str or mapping, default "bce_logits"
            Binary loss applied to target-cluster masks.
        seed_loss : str or mapping, default "l1"
            Regression loss applied to seediness predictions.
        embedding_weight : float, default 1
            Weight of the target-cluster mask loss.
        seediness_weight : float, default 1
            Weight of the seediness regression loss.
        smoothing_weight : float, default 1
            Weight penalizing margin variation within a target cluster.
        inter_weight : float, default 1
            Weight penalizing target centroids that are too close.
        inter_margin : float, default 0.2
            Minimum centroid-separation scale in embedding space.
        min_voxels : int, default 2
            Minimum number of voxels required to supervise one semantic class.
        eps : float, default 1e-6
            Numerical stability constant.
        """
        # Validate scalar loss parameters before constructing objectives
        weights = {
            "embedding_weight": embedding_weight,
            "seediness_weight": seediness_weight,
            "smoothing_weight": smoothing_weight,
            "inter_weight": inter_weight,
        }
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("SPICE loss weights must be nonnegative.")
        if inter_margin < 0.0:
            raise ValueError("`inter_margin` must be nonnegative.")
        if min_voxels < 1:
            raise ValueError("`min_voxels` must be positive.")
        if eps <= 0.0:
            raise ValueError("`eps` must be positive.")

        # Initialize component objectives and store their relative weights
        self.mask_loss_fn = loss_fn_factory(mask_loss, reduction="none")
        self.seed_loss_fn = loss_fn_factory(seed_loss, reduction="mean")
        self.embedding_weight = embedding_weight
        self.seediness_weight = seediness_weight
        self.smoothing_weight = smoothing_weight
        self.inter_weight = inter_weight
        self.inter_margin = inter_margin
        self.min_voxels = min_voxels
        self.eps = eps

    @staticmethod
    def _cluster_means(
        values: torch.Tensor,
        labels: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """Compute one mean feature vector per contiguous cluster label."""
        # Accumulate cluster sums and normalize by their voxel multiplicities
        sums = values.new_zeros((num_clusters, values.shape[1]))
        sums.index_add_(0, labels, values)
        counts = torch.bincount(labels, minlength=num_clusters).to(values.dtype)
        return sums / counts[:, None]

    def _semantic_loss(
        self,
        embeddings: torch.Tensor,
        margins: torch.Tensor,
        seediness: torch.Tensor,
        cluster_labels: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int] | None
    ):
        """Compute SPICE terms for one batch entry and semantic class."""
        # Remove invalid cluster assignments and undersized supervision sets
        valid = cluster_labels >= 0
        if int(valid.sum()) < self.min_voxels:
            return None

        embeddings = embeddings[valid]
        margins = margins[valid]
        seediness = seediness[valid]
        cluster_labels = cluster_labels[valid]
        _, labels = torch.unique(
            cluster_labels,
            sorted=True,
            return_inverse=True,
        )
        num_clusters = int(labels.max()) + 1

        # Estimate one embedding centroid and Gaussian width per cluster
        centroids = self._cluster_means(embeddings, labels, num_clusters)
        cluster_margins = self._cluster_means(margins, labels, num_clusters)
        cluster_margins = torch.clamp(cluster_margins[:, 0], min=self.eps)

        # Convert embedding distances into cluster-membership probabilities
        squared_distances = (
            (embeddings[:, None, :] - centroids[None, :, :]).square().sum(dim=2)
        )
        probabilities = torch.exp(
            -squared_distances / (2.0 * cluster_margins[None, :].square())
        )
        probabilities = torch.clamp(
            probabilities,
            min=self.eps,
            max=1.0 - self.eps,
        )
        logits = torch.logit(probabilities)
        targets = torch.nn.functional.one_hot(
            labels,
            num_classes=num_clusters,
        ).to(embeddings.dtype)

        # Supervise masks, seediness, and within-cluster margin consistency
        mask_loss = self.mask_loss_fn(logits, targets).mean()
        own_probabilities = probabilities.gather(1, labels[:, None]).flatten()
        seed_loss = self.seed_loss_fn(
            seediness.flatten(),
            own_probabilities.detach(),
        ).mean()
        smoothing_loss = torch.abs(
            margins[:, 0] - cluster_margins[labels].detach()
        ).mean()

        # Penalize distinct cluster centroids that violate the separation margin
        inter_loss = embeddings.sum() * 0.0
        if num_clusters > 1:
            centroid_distances = torch.pdist(centroids)
            inter_loss = (
                torch.clamp(
                    2.0 * self.inter_margin - centroid_distances,
                    min=0.0,
                )
                .square()
                .mean()
            )

        # Measure the mean mask intersection-over-union at the default threshold
        predictions = probabilities >= 0.5
        target_masks = targets.bool()
        intersection = (predictions & target_masks).sum(dim=0).float()
        union = (predictions | target_masks).sum(dim=0).float()
        accuracy = float((intersection / union).mean())

        return (
            mask_loss,
            seed_loss,
            smoothing_loss,
            inter_loss,
            accuracy,
            len(cluster_labels),
        )

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        embeddings: TensorBatch,
        margins: TensorBatch,
        seediness: TensorBatch,
        filter_index: IndexBatch,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor | float | int]:
        """Compute the SPICE loss for one filtered batch.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Original voxel-wise cluster labels.
        embeddings : TensorBatch
            ``(M, D)`` filtered spatial embeddings.
        margins : TensorBatch
            ``(M, 1)`` positive cluster-margin predictions.
        seediness : TensorBatch
            ``(M, 1)`` seediness predictions.
        filter_index : IndexBatch
            ``(M,)`` mapping from filtered voxels to original input rows.
        **kwargs : dict, optional
            Other upstream outputs unused by this loss.

        Returns
        -------
        dict
            Total loss, component losses, mean mask IoU, and supervised voxel
            count.
        """
        # Validate and unwrap the filtered network products
        embedding_tensor = embeddings.torch_tensor()
        margin_tensor = margins.torch_tensor()
        seediness_tensor = seediness.torch_tensor()
        if not (
            len(embedding_tensor)
            == len(margin_tensor)
            == len(seediness_tensor)
            == len(filter_index.index)
        ):
            raise ValueError("Filtered SPICE outputs and index must have equal length.")

        # Recover the corresponding truth rows from the original label batch
        shape_tensor = clust_label.shapes.torch_tensor()[filter_index.index]
        cluster_tensor = clust_label.cluster_ids.torch_tensor()[filter_index.index]
        terms = []
        supervised_voxels = 0

        # Evaluate each event and semantic class as an independent mask problem
        for batch_id in range(embeddings.batch_size):
            lower = int(embeddings.edges[batch_id])
            upper = int(embeddings.edges[batch_id + 1])
            embeddings_b = embedding_tensor[lower:upper]
            margins_b = margin_tensor[lower:upper]
            seediness_b = seediness_tensor[lower:upper]
            shapes_b = shape_tensor[lower:upper].long()
            clusters_b = cluster_tensor[lower:upper].long()

            for shape in torch.unique(shapes_b):
                shape_mask = shapes_b == shape
                semantic_terms = self._semantic_loss(
                    embeddings_b[shape_mask],
                    margins_b[shape_mask],
                    seediness_b[shape_mask],
                    clusters_b[shape_mask],
                )
                if semantic_terms is not None:
                    terms.append(semantic_terms)
                    supervised_voxels += semantic_terms[5]

        # Preserve a differentiable zero when no class can be supervised
        if not terms:
            zero = embedding_tensor.sum() * 0.0
            return {
                "loss": zero,
                "mask_loss": 0.0,
                "seed_loss": 0.0,
                "smoothing_loss": 0.0,
                "inter_loss": 0.0,
                "accuracy": 1.0,
                "count": 0,
            }

        # Average component terms across supervised semantic subsets
        mask_loss = torch.stack([term[0] for term in terms]).mean()
        seed_loss = torch.stack([term[1] for term in terms]).mean()
        smoothing_loss = torch.stack([term[2] for term in terms]).mean()
        inter_loss = torch.stack([term[3] for term in terms]).mean()

        # Form the weighted objective and aggregate mask accuracy
        loss = (
            self.embedding_weight * mask_loss
            + self.seediness_weight * seed_loss
            + self.smoothing_weight * smoothing_loss
            + self.inter_weight * inter_loss
        )
        accuracy = sum(term[4] for term in terms) / len(terms)

        return {
            "loss": loss,
            "mask_loss": float(mask_loss.detach()),
            "seed_loss": float(seed_loss.detach()),
            "smoothing_loss": float(smoothing_loss.detach()),
            "inter_loss": float(inter_loss.detach()),
            "accuracy": accuracy,
            "count": supervised_voxels,
        }
