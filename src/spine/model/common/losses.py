"""Custom loss functions shared by multiple model families."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import torch

from spine.model.common.weighting import get_class_weights

from .act_norm import act_factory

__all__ = [
    "LogRMSE",
    "BerHuLoss",
    "BinaryDiceLoss",
    "BinaryLogDiceLoss",
    "BinaryMincutLoss",
    "BinaryLogDiceCELoss",
    "BinaryLogDiceCEMincutLoss",
    "BinaryFocalLoss",
]

Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> Reduction:
    """Validate and narrow a loss reduction name.

    Parameters
    ----------
    reduction : str
        Reduction requested by the caller.

    Returns
    -------
    {"none", "mean", "sum"}
        Validated reduction name.

    Raises
    ------
    ValueError
        If the reduction is not supported.
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            f"Reduction must be one of 'none', 'mean' or 'sum', got '{reduction}'."
        )
    return reduction


def _reduce_loss(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """Apply a standard reduction to an element-wise loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class LogRMSE(torch.nn.Module):
    """Compute point-wise root squared error in logarithmic space."""

    name = "log_rmse"

    def __init__(self, reduction: Reduction = "none", eps: float = 1e-7) -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        reduction : str, default 'none'
            Reduction function to apply to the output
        eps : float, default 1e-7
            Offset to apply to the predictions/labels before passing them
            through the MSE loss function.
        """
        super().__init__()

        if eps <= 0:
            raise ValueError("`eps` must be strictly positive.")

        self.reduction: Reduction = _validate_reduction(reduction)
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the logarithmic root squared error.

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
        if torch.any(inputs + self.eps <= 0) or torch.any(targets + self.eps <= 0):
            raise ValueError("LogRMSE inputs and targets must be greater than `-eps`.")

        x = torch.log(inputs + self.eps)
        y = torch.log(targets + self.eps)
        loss = torch.sqrt((x - y).square() + self.eps)
        return _reduce_loss(loss, self.reduction)


class BerHuLoss(torch.nn.Module):
    """Compute the reverse Huber (BerHu) regression loss.

    Residuals below a batch-dependent threshold use an L1 penalty, while
    larger residuals use a smooth quadratic penalty.
    """

    name = "berhu"

    def __init__(self, threshold: float = 0.2, reduction: Reduction = "none") -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        threshold : float, default 0.2
            Fraction of the maximum loss value to use as a threshold
        reduction : str, default 'none'
            Reduction function to apply to the output
        """
        super().__init__()

        if not 0 < threshold <= 1:
            raise ValueError("`threshold` must lie in the interval (0, 1].")

        self.threshold = threshold
        self.reduction: Reduction = _validate_reduction(reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the BerHu loss.

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
        norm = torch.abs(inputs - targets)

        if norm.numel() == 0:
            return _reduce_loss(norm, self.reduction)

        c = norm.max() * self.threshold
        if c == 0:
            return _reduce_loss(norm, self.reduction)

        out = torch.where(norm <= c, norm, (norm**2 + c**2) / (2.0 * c))
        return _reduce_loss(out, self.reduction)


class BinaryDiceLoss(torch.nn.Module):
    """Applies the binary Dice Loss.

    The Dice loss is derived from the Dice Similarity Coefficient, also known
    as the Sorensen–Dice coefficient, which is a statistical measure used to
    compare the similarity of two samples.
    """

    name = "binary_dice"

    def __init__(
        self,
        eps: float = 1e-6,
        squared_pred: bool = True,
        activation: str | Mapping[str, Any] = "sigmoid",
    ) -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        eps : float, default 1e-6
            Regularization constant for the ratio
        squared_pred : bool, default True
            Whether to square probabilities and targets in the denominator
        activation : str or mapping, default 'sigmoid'
            Activation configuration applied to the input logits
        """
        super().__init__()

        if eps <= 0:
            raise ValueError("`eps` must be strictly positive.")

        self.eps = eps
        self.squared_pred = squared_pred
        self.act = act_factory(activation)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the binary Dice loss.

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
        probas = self.act(logits)
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the binary logarithmic Dice loss.

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


class BinaryMincutLoss(torch.nn.Module):
    """Compute a simple binary minimum-cut objective.

    The loss is one minus the summed overlap between predicted probabilities
    and the target mask.
    """

    name = "binary_mincut"

    def __init__(self, activation: str | Mapping[str, Any] = "sigmoid") -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        activation : str or mapping, default 'sigmoid'
            Activation configuration applied to the input logits
        """
        super().__init__()
        self.act = act_factory(activation)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the minimum-cut loss.

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
        probas = self.act(logits)
        return 1.0 - (probas * targets).sum()


class BinaryLogDiceCELoss(torch.nn.Module):
    """Applies the binary log Dice loss and the cross-entropy loss.

    This class inherits from the :class:`BinaryLogDiceLoss` and adds
    a cross-entropy loss on top of it, with some configurable weights.
    """

    name = "binary_log_dice_ce"

    def __init__(
        self,
        log_dice: Mapping[str, Any] | None = None,
        bce: Mapping[str, Any] | None = None,
        reduction: Reduction = "mean",
        w_dice: float = 0.8,
        w_ce: float = 0.2,
    ) -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        log_dice : dict, optional
            Parameters to pass to the :class:`BinaryLogDiceLoss`
        bce : dict, optional
            Parameters to pass to the :class:`torch.nn.BCEWithLogitsLoss`
        reduction : str, default 'mean'
            Reduction function to apply to the BCE loss
        w_dice : float, default 0.8
            Prefactor applied to the log Dice loss
        w_ce : float, default 0.2
            Prefactor to be applied to the binary cross-entropy loss
        """
        super().__init__()

        log_dice_config = {} if log_dice is None else dict(log_dice)
        self.log_dice = BinaryLogDiceLoss(**log_dice_config)

        bce_config = {} if bce is None else dict(bce)
        if "reduction" in bce_config:
            raise ValueError(
                "Specify BCE reduction through the top-level `reduction` argument."
            )
        self.reduction: Reduction = _validate_reduction(reduction)
        self.bce = torch.nn.BCEWithLogitsLoss(**bce_config, reduction=self.reduction)

        self.w_dice = w_dice
        self.w_ce = w_ce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted logarithmic Dice and BCE loss.

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
        log_dice = self.log_dice(logits, targets)
        bce = self.bce(logits, targets.float())
        return self.w_dice * log_dice + self.w_ce * bce


class BinaryLogDiceCEMincutLoss(BinaryLogDiceCELoss):
    """Applies the binary log Dice loss, cross-entropy loss and mincut loss.

    This class inherits from the :class:`BinaryLogDiceCELoss` and adds
    a mincut loss on top of it, with some configurable weights.
    """

    name = "binary_log_dice_ce_mincut"

    def __init__(
        self,
        mincut: Mapping[str, Any] | None = None,
        w_mincut: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the loss function parameters.

        Parameters
        ----------
        mincut : dict, optional
            Parameters to pass to :class:`BinaryMincutLoss`
        w_mincut : float, default 1.0
            Prefactor applied to the min-cut loss
        **kwargs : dict, optional
            Parameters to pass to the :class:`BinaryLogDiceCELoss`
        """
        super().__init__(**kwargs)

        mincut_config = {} if mincut is None else dict(mincut)
        self.mincut = BinaryMincutLoss(**mincut_config)

        # Check that the activation functions are consistent
        if type(self.log_dice.act) is not type(self.mincut.act):
            raise ValueError(
                "The log Dice loss and Mincut loss must have the same "
                "activation functions."
            )

        self.w_mincut = w_mincut

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted Dice, BCE and minimum-cut loss.

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
        log_dice = self.log_dice(logits, targets)
        bce = self.bce(logits, targets.float())
        mincut = self.mincut(logits, targets)
        return self.w_dice * log_dice + self.w_ce * bce + self.w_mincut * mincut


class BinaryFocalLoss(torch.nn.Module):
    """Compute binary focal loss.

    This implementation follows the focal-loss modulation proposed in
    https://arxiv.org/abs/1708.02002.
    """

    name = "binary_focal"

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        logits: bool = False,
        balance_loss: bool = False,
        reduction: Reduction = "none",
    ) -> None:
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
        super().__init__()

        if alpha < 0:
            raise ValueError("`alpha` must be nonnegative.")
        if gamma < 0:
            raise ValueError("`gamma` must be nonnegative.")

        self.alpha = alpha
        self.gamma = gamma
        self.balance_loss = balance_loss
        self.reduction: Reduction = _validate_reduction(reduction)

        if logits:
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.bce = torch.nn.BCELoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the binary focal loss.

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
        bce = self.bce(inputs, targets.float())
        pt = torch.exp(-bce)
        out = self.alpha * (1 - pt) ** self.gamma * bce

        if self.balance_loss:
            with torch.no_grad():
                weights = get_class_weights(targets.long(), 2, per_class=False)
            out = out * weights

        return _reduce_loss(out, self.reduction)
