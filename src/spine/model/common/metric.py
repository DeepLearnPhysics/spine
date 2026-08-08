"""Modules used to evaluate model performance."""

from __future__ import annotations

import torch

__all__ = ["IoUScore"]


class IoUScore(torch.nn.Module):
    """Intersection over union score for binary predictions."""

    name = "iou"

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Evaluate the IoU score for a batch of label and predictions.

        Parameters
        ----------
        y_true : torch.Tensor
            Set of labels
        y_pred : torch.Tensor
            Set of predictions

        Returns
        -------
        float
            IoU score
        """
        with torch.no_grad():
            union = (y_true.long() == 1) | (y_pred.long() == 1)
            if not union.any():
                return 0.0

            intersection = (y_true.long() == 1) & (y_pred.long() == 1)
            return float(intersection.sum() / union.sum())
