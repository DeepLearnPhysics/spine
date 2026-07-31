"""Binary edge-classification loss for GraphSPICE."""

from __future__ import annotations

from typing import cast

import torch

from spine.data import TensorBatch
from spine.model.common.factories import loss_fn_factory, metric_fn_factory
from spine.utils.factory import Config
from spine.utils.weighting import get_class_weights

__all__ = ["EdgeLoss"]


class BinaryMetric(torch.nn.Module):
    """Typing contract for binary metrics built by the shared factory."""

    name: str


class EdgeLoss(torch.nn.Module):
    """Supervise the binary edge logits produced by GraphSPICE."""

    name = "edge"

    def __init__(
        self,
        loss: Config = "binary_log_dice_ce",
        invert: bool = True,
        balance_loss: bool = False,
        equal_sampling: bool = False,
        min_sample_edges: int = 1000,
        metric: Config | None = "iou",
    ) -> None:
        """Initialize the edge-classification loss.

        Parameters
        ----------
        loss : str or mapping, default "binary_log_dice_ce"
            Binary loss configuration.
        invert : bool, default True
            Treat disconnected edges as the positive class.
        balance_loss : bool, default False
            Weight individual edges to balance the two target classes.
        equal_sampling : bool, default False
            Draw the same number of examples from each target class.
        min_sample_edges : int, default 1000
            Minimum number of examples drawn from each class when equal
            sampling is enabled. Sampling uses replacement when necessary.
        metric : str or mapping, optional
            Additional binary metric configuration.
        """
        super().__init__()

        if min_sample_edges < 1:
            raise ValueError("`min_sample_edges` must be positive.")

        self.invert = invert
        self.balance_loss = balance_loss
        self.equal_sampling = equal_sampling
        self.min_sample_edges = min_sample_edges
        self.loss_fn = loss_fn_factory(loss, reduction="none")
        self.metric_fn = (
            None if metric is None else cast(BinaryMetric, metric_fn_factory(metric))
        )

    def sample_edges(
        self,
        edge_logits: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw an equal number of edges from both target classes.

        Parameters
        ----------
        edge_logits : torch.Tensor
            ``(E,)`` edge-logit predictions.
        edge_labels : torch.Tensor
            ``(E,)`` binary edge labels.

        Returns
        -------
        tuple of torch.Tensor
            Sampled logits and corresponding labels. If either class is
            absent, the original tensors are returned unchanged.
        """
        class_indices = [
            torch.where(edge_labels == class_id)[0] for class_id in range(2)
        ]
        if any(len(indices) == 0 for indices in class_indices):
            return edge_logits, edge_labels

        sample_count = max(
            min(len(indices) for indices in class_indices),
            self.min_sample_edges,
        )
        sampled_indices = []
        for indices in class_indices:
            if sample_count <= len(indices):
                permutation = torch.randperm(
                    len(indices),
                    device=indices.device,
                )[:sample_count]
                sampled_indices.append(indices[permutation])
            else:
                selection = torch.randint(
                    len(indices),
                    (sample_count,),
                    device=indices.device,
                )
                sampled_indices.append(indices[selection])

        index = torch.cat(sampled_indices)
        return edge_logits[index], edge_labels[index]

    def forward(
        self,
        clust_label: TensorBatch,
        edge_attr: TensorBatch,
        edge_label: TensorBatch,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Compute the loss and metrics for one batch of graph edges.

        Parameters
        ----------
        clust_label : TensorBatch
            Voxel cluster labels. Present for the shared loss interface and
            unused by this loss.
        edge_attr : TensorBatch
            ``(E,)`` edge-logit predictions.
        edge_label : TensorBatch
            ``(E,)`` binary edge labels.
        **kwargs : object, optional
            Additional upstream outputs unused by this loss.

        Returns
        -------
        dict
            Loss, binary accuracy, number of supervised edges, and the
            configured optional metric.
        """
        edge_logits = edge_attr.torch_tensor().flatten()
        edge_labels = edge_label.torch_tensor().flatten()
        if len(edge_logits) != len(edge_labels):
            raise ValueError(
                "Edge logits and labels must have the same length, got "
                f"{len(edge_logits)} and {len(edge_labels)}."
            )
        if torch.any((edge_labels != 0) & (edge_labels != 1)):
            raise ValueError("Edge labels must be binary (0 or 1).")
        edge_labels = edge_labels.long()

        if self.equal_sampling:
            edge_logits, edge_labels = self.sample_edges(
                edge_logits,
                edge_labels,
            )

        if self.invert:
            edge_labels = torch.logical_not(edge_labels).long()

        edge_predictions = (edge_logits > 0.0).long()
        num_edges = len(edge_predictions)

        # Preserve a differentiable zero for empty graphs instead of reducing
        # an empty tensor to NaN.
        if num_edges == 0:
            loss = edge_logits.sum() * 0.0
        else:
            loss = self.loss_fn(edge_logits, edge_labels.float())
            if self.balance_loss:
                weights = get_class_weights(
                    edge_labels,
                    num_classes=2,
                    per_class=False,
                )
                loss *= weights
            loss = loss.mean()

        accuracy = 1.0
        if num_edges > 0:
            accuracy = float((edge_predictions == edge_labels).sum() / num_edges)

        metric = {}
        if self.metric_fn is not None:
            metric[self.metric_fn.name] = self.metric_fn(
                edge_labels,
                edge_predictions,
            )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "count": num_edges,
            **metric,
        }
