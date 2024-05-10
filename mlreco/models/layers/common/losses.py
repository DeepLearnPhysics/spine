"""Module with losses which are not generically provided by `PyTorch`."""

import torch
from torch import nn

__all__ = ['LogRMSE', 'BerHuLoss']


class LogRMSE(nn.modules.loss._Loss):
    """Applies RMSE loss to in the log space for regression tasks."""
    name = 'log_rmse'

    def __init__(self, reduction='none', eps=1e-7):
        """Initialize the loss function parameters.

        Parameters
        ----------
        reduction : str, default 'none'
            Reduction function to apply to the output
        eps : float, default 1e-7
            Offset to apply to the predictions/labels before passing them
            through the MSE loss function.
        """
        # Initialize the parent class
        super().__init__()

        # Store the attributes
        self.reduction = reduction
        self.eps = eps

        # Initialize the underlying MSE loss function
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            Values predicted by the network
        targets : torch.Tensor
            Regression targets

        Returns
        -------
        torch.Tensor
            Loss value or array of loss values (if no reduction)
        """
        # Move the input/target to the log space
        x = torch.log(inputs + self.eps)
        y = torch.log(targets + self.eps)

        # Compute the RMSE loss
        out = self.mseloss(x, y)
        out = torch.sqrt(out + self.eps)

        # Return the appropriate reduction output
        if self.reduction == 'none':
            return out
        elif self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            raise ValueError(
                    "Reduction function not recognized:", self.reduction)


class BerHuLoss(nn.modules.loss._Loss):
    """Applies the BerHu loss."""
    name = 'log_rmse'

    def __init__(self, threshold=0.2, reduction='none'):
        """Initialize the loss function parameters.

        Parameters
        ----------
        threshold : float, default 0.2
            Fraction of the maximum loss value to use as a threshold
        reduction : str, default 'none'
            Reduction function to apply to the output
        """
        # Initialize the parent class
        super().__init__()

        # Store the attributes
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            Values predicted by the network
        targets : torch.Tensor
            Regression targets

        Returns
        -------
        torch.Tensor
            Loss value or array of loss values (if no reduction)
        """
        # Compute the L1 loss
        norm = torch.abs(inputs - targets)

        # If the norm array is of length 0, nothing to do
        if len(norm) == 0:
            return norm.sum()

        # Apply different losses below and above the threshold
        c = norm.max() * self.threshold
        out = torch.where(norm <= c, norm, (norm**2 + c**2) / (2.0 * c))

        # Return the appropriate reduction output
        if self.reduction == 'none':
            return out
        elif self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            raise ValueError(
                    "Reduction function not recognized:", self.reduction)
