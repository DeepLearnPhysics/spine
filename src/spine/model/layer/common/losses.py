"""Module with losses which are not generically provided by `PyTorch`."""

import torch
from torch import nn

from spine.utils.weighting import get_class_weights

from .act_norm import act_factory

__all__ = [
    "LogRMSE",
    "BerHuLoss",
    "BinaryDiceLoss",
    "BinaryLogDiceLoss",
    "BinaryMincutLoss",
    "BinaryLogDiceCELoss",
    "BinaryLogDiceCEMincutLoss",
    "FocalLoss",
]


class LogRMSE(nn.modules.loss._Loss):
    """Applies RMSE loss to in the log space for regression tasks."""

    name = "log_rmse"

    def __init__(self, reduction="none", eps=1e-7):
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
        self.mseloss = nn.MSELoss(reduction="none")

    def forward(self, inputs, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

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
        if self.reduction == "none":
            return out
        elif self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            raise ValueError("Reduction function not recognized:", self.reduction)


class BerHuLoss(nn.modules.loss._Loss):
    """Applies the BerHu loss."""

    name = "berhu"

    def __init__(self, threshold=0.2, reduction="none"):
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
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

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
        if self.reduction == "none":
            return out
        elif self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            raise ValueError("Reduction function not recognized:", self.reduction)


class BinaryDiceLoss(nn.modules.loss._Loss):
    """Applies the binary Dice Loss.

    The Dice loss is derived from the Dice Similarity Coefficient, also known
    as the Sorensen–Dice coefficient, which is a statistical measure used to
    compare the similarity of two samples.
    """

    name = "binary_dice"

    def __init__(self, eps=1e-6, squared_pred=True, activation="sigmoid"):
        """Initialize the loss function parameters.

        Parameters
        ----------
        eps : float, default 1e-6
            Regularization constant for the ratio
        """
        # Initialize the parent class
        super().__init__()

        # Store the loss parameters
        self.eps = eps
        self.squared_pred = squared_pred

        # Intitialize the activation layer
        self.act = act_factory(activation)

    def forward(self, logits, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Compute probability measures using the activation function
        probas = self.act(logits)

        # Compute the dice loss
        inter = (probas * targets).sum()
        if not self.squared_pred:
            den = probas.sum() + targets.sum()
        else:
            den = (probas**2).sum() + (targets**2).sum()

        return 1.0 - (2 * inter + self.eps) / (den + self.eps)


class BinaryLogDiceLoss(BinaryDiceLoss):
    """Applies the binary log Dice loss.

    This class inherits from the standard :class:`BinaryDiceLoss` and simply
    passes it through a logarithm.
    """

    name = "binary_log_dice"

    def forward(self, logits, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        dice = super().forward(logits, targets)
        dice = torch.clamp(dice, min=self.eps, max=1.0 - self.eps)
        return -torch.log(1.0 - dice)


class BinaryMincutLoss(nn.modules.loss._Loss):
    """Applies the min-cut loss.

    This is a very basic loss of 1. - interesection between the output
    probabilities and the target domain.
    """

    name = "binary_mincut"

    def __init__(self, activation="sigmoid"):
        """Initialize the loss function parameters.

        Parameters
        ----------
        eps : float, default 1e-6
            Regularization constant for the ratio
        """
        # Initialize the parent class
        super().__init__()

        # Intitialize the activation layer
        self.act = act_factory(activation)

    def forward(self, logits, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Compute probability measures using the activation function
        probas = self.act(logits)

        # Compute the mincut loss
        return 1.0 - (probas * targets).sum()


class BinaryLogDiceCELoss(nn.modules.loss._Loss):
    """Applies the binary log Dice loss and the cross-entropy loss.

    This class inherits from the :class:`BinaryLogDiceLoss` and adds
    a cross-entropy loss on top of it, with some configurable weights.
    """

    name = "binary_log_dice_ce"

    def __init__(self, log_dice=None, bce=None, reduction="mean", w_dice=0.8, w_ce=0.2):
        """Initialize the loss function parameters.

        Parameters
        ----------
        log_dice : dict, optional
            Parameters to pass to the :class:`BinaryLogDiceLoss`
        bce : dict, optional
            Parameters to pass to the :class:`torch.nn.BCEWithLogitsLoss`
        reduction : str, default 'mean'
            Reduction function to apply tot he BCE loss
        w_dice : float, default 0.8
            Prefacor to be applied to the log Dice loss
        w_ce : float, default 0.2
            Prefactor to be applied to the binary cross-entropy loss
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the binary log DICE loss
        log_dice = (log_dice is not None) or {}
        self.log_dice = BinaryLogDiceLoss(**log_dice)

        # Initiliaze the binary cross-entropy loss
        bce = (bce is not None) or {}
        self.bce = torch.nn.BCEWithLogitsLoss(**bce, reduction=reduction)

        # Store the loss component weights
        self.w_dice = w_dice
        self.w_ce = w_ce

    def forward(self, logits, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Compute the log dice loss
        log_dice = self.log_dice(logits, targets)

        # Compute the mean binary cross-entropy loss
        bce = self.bce(logits, targets.float())

        # Combine the losses
        return self.w_dice * log_dice + self.w_ce * bce


class BinaryLogDiceCEMincutLoss(nn.modules.loss._Loss):
    """Applies the binary log Dice loss, cross-entropy loss and mincut loss.

    This class inherits from the :class:`BinaryLogDiceCELoss` and adds
    a mincut loss on top of it, with some configurable weights.
    """

    name = "binary_log_dice_ce_mincut"

    def __init__(self, w_mincut=None, **kwargs):
        """Initialize the loss function parameters.

        Parameters
        ----------
        w_mincut : float, default 1.0
            Prefacor to be applied to the Mincut loss
        mincut : dict, optional
            Parameters to pass to the :class:`Mincut`
        **kwargs : dict, optional
            Parameters to pass to the :class:`BinaryLogDiceCELoss`
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Initiliaze the binary Mincut loss
        mincut = (mincut is not None) or {}
        self.mincut = MincutLoss(**mincut)

        # Check that the activation functions are consistent
        assert self.log_dice.act == self.mincut.act, (
            "The log Dice loss and Mincut loss must have the same "
            "activation functions."
        )

        # Store the loss component weights
        self.w_mincut = w_mincut

    def forward(self, logits, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Compute the log dice loss
        log_dice = self.log_dice(logits, targets)

        # Compute the mean binary cross-entropy loss
        bce = self.bce(logits, targets.float())

        # Compute the mincut loss
        mincut = self.mincut(logits, targets)

        # Combine the losses
        return self.w_dice * log_dice + self.w_ce * bce + self.w_mincut * mincut


class BinaryFocalLoss(nn.modules.loss._Loss):
    """Applies the focal loss.

    Original Paper: https://arxiv.org/abs/1708.02002
    """

    name = "binary_focal"

    def __init__(
        self, alpha=1, gamma=2, logits=False, balance_loss=False, reduction=True
    ):
        """Initialize the loss function parameters.

        Parameters
        ----------
        alpha : float, default 1
            Overall loss scaling factor
        gamma : float, default 2
            Overall power to apply to the score prefactor
        logits : bool, default False
            If `True`, the output of the network is considered to be logits
        balance_loss : bool, default False
            If `True`, weights are applied to the loss to account for class imbalance
        reduction : str, default 'none'
            Reduction function to apply to the output
        """
        # Initialize the parent class
        super().__init__()

        # Store the loss parameters
        self.alpha = alpha
        self.gamma = gamma
        self.balance_loss = balance_loss
        self.reduction = reduction

        # Initialize the BCE layer
        if logits:
            self.bce = torch.nn.BCELoss(reduction="none")
        else:
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        """Pass predictions/labels through the loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            (N) Values predicted by the network
        targets : torch.Tensor
            (N) Regression targets

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Compute the BCE loss
        bce = self.bce(inputs, targets.float())

        # Compute cross-entropy loss to softmax scores, compute focal lsos
        pt = torch.exp(-bce)
        out = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # If requested, balance classes
        if self.balance_loss:
            with torch.no_grad():
                weights = get_class_weights(targets, 2, per_class=False)

        # Return the appropriate reduction output
        if self.reduction == "none":
            return out
        elif self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            raise ValueError("Reduction function not recognized:", self.reduction)
