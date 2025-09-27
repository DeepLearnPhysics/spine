"""Modules used to evaluate model performance."""

import torch

__all__ = ["IoUScore"]


class IoUScore(torch.nn.Module):
    """Intersection over union score for binary predictions."""

    name = "iou"

    def forward(self, y_true, y_pred):
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
        # Compute and return
        with torch.no_grad():
            union = (y_true.long() == 1) | (y_pred.long() == 1)
            if not union.any():
                return 0.0

            else:
                intersection = (y_true.long() == 1) & (y_pred.long() == 1)
                return float(intersection.sum() / union.sum())
