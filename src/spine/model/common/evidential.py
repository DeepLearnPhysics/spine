"""Evidential prediction heads and classification/regression objectives."""

from __future__ import annotations

from collections.abc import Callable
from math import log, pi
from typing import Any, Literal

import torch

from .mlp import MLP

__all__ = ["EvidentialModel", "EVDLoss", "EDLRegressionLoss"]

Reduction = Literal["none", "mean", "sum"]


class EvidentialModel(torch.nn.Module):
    """Predict normal-inverse-gamma parameters with an MLP backbone.

    The four returned columns are ``gamma``, ``nu``, ``alpha`` and ``beta``.
    The latter three are constrained to the valid normal-inverse-gamma
    parameter domain.
    """

    def __init__(
        self,
        in_channels: int,
        mlp: dict[str, Any],
        eps: float = 0.0,
        logspace: bool = False,
    ) -> None:
        """Initialize the evidential network.

        Parameters
        ----------
        in_channels : int
            Number of features from the upstream feature extractor
        mlp : dict
            MLP configuration dictionary
        eps : float, default 0.0
            Offset to apply to the softplus output
        logspace : bool, default False
            Whether the regression target and predicted mean use log space.

        Raises
        ------
        ValueError
            If ``eps`` is negative.
        """
        super().__init__()

        if eps < 0.0:
            raise ValueError(f"`eps` must be nonnegative, got {eps}.")

        self.mlp = MLP(in_channels, **mlp)
        self.linear = torch.nn.Linear(self.mlp.feature_size, 4)

        self.eps = eps
        self.softplus = torch.nn.Softplus()
        self.logspace = logspace
        self.gamma = torch.nn.Sigmoid() if logspace else torch.nn.Identity()

    def forward(self, input_feats: torch.Tensor) -> torch.Tensor:
        """Convert input features into valid evidential parameters.

        Parameters
        ----------
        input_feats : torch.Tensor
            ``(N, F)`` tensor of input features.

        Returns
        -------
        torch.Tensor
            ``(N, 4)`` tensor containing ``gamma``, ``nu``, ``alpha`` and
            ``beta``.
        """
        logits = self.linear(self.mlp(input_feats))

        positive_parameters = self.softplus(logits[:, :3]) + self.eps
        nu = positive_parameters[:, 0].view(-1, 1)
        alpha = torch.clamp(
            positive_parameters[:, 1] + 1.0,
            min=1.0,
        ).view(-1, 1)
        beta = positive_parameters[:, 2].view(-1, 1)
        gamma = 2.0 * self.gamma(logits[:, 3]).view(-1, 1)

        evidence = torch.cat((gamma, nu, alpha, beta), dim=1)
        if not self.logspace:
            evidence = torch.clamp(evidence, min=self.eps)

        return evidence


def _reduce_loss(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """Apply a standard reduction to a loss tensor.

    Parameters
    ----------
    loss : torch.Tensor
        Unreduced loss values.
    reduction : {"none", "mean", "sum"}
        Reduction applied to ``loss``.

    Returns
    -------
    torch.Tensor
        Reduced or unreduced loss.

    Raises
    ------
    ValueError
        If ``reduction`` is unknown.
    """
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction method `{reduction}`.")


def digamma_evd_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute Dirichlet Bayes risk using the digamma formulation.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(N, C)`` positive Dirichlet concentration parameters.
    targets : torch.Tensor
        ``(N, C)`` one-hot or probabilistic class targets.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced loss.
    """
    total_concentration = alpha.sum(dim=1, keepdim=True)
    loss = torch.sum(
        (torch.digamma(total_concentration) - torch.digamma(alpha)) * targets,
        dim=1,
    )
    return loss


def sumsq_evd_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute squared-error Bayes risk for a Dirichlet prediction.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(N, C)`` positive Dirichlet concentration parameters.
    targets : torch.Tensor
        ``(N, C)`` one-hot or probabilistic class targets.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced loss.
    """
    total_concentration = alpha.sum(dim=1, keepdim=True)
    prediction_err = (targets - alpha / total_concentration) ** 2
    variance = (
        alpha
        * (total_concentration - alpha)
        / (total_concentration * total_concentration * (total_concentration + 1.0))
    )
    loss = torch.sum(prediction_err + variance, dim=1)
    return loss


def nll_evd_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the expected negative log-likelihood under a Dirichlet.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(N, C)`` positive Dirichlet concentration parameters.
    targets : torch.Tensor
        ``(N, C)`` one-hot or probabilistic class targets.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced loss.
    """
    total_concentration = alpha.sum(dim=1, keepdim=True)
    loss = torch.sum(
        targets * (torch.log(total_concentration) - torch.log(alpha)),
        dim=1,
    )
    return loss


def evd_kl_divergence(
    alpha: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KL divergence between two Dirichlet distributions.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(N, C)`` concentration parameters for the predicted distribution.
    beta : torch.Tensor, optional
        Broadcastable reference concentration parameters. Defaults to the
        uniform Dirichlet distribution.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced KL divergence.
    """
    alpha_total = torch.sum(alpha, dim=1)
    if beta is None:
        beta = alpha.new_ones((1, alpha.shape[1]))
    beta_total = torch.sum(beta, dim=1)
    loss = torch.lgamma(alpha_total) - torch.lgamma(beta_total)
    loss -= torch.sum(torch.lgamma(alpha), dim=1)
    divergence_terms = (alpha - beta) * (
        torch.digamma(alpha) - torch.digamma(alpha_total.view(-1, 1))
    )
    loss += torch.sum(divergence_terms, dim=1)
    return loss


def evd_loss_dict() -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Build the registry of evidential classification risk functions.

    Returns
    -------
    dict
        Mapping from configuration names to unreduced loss functions.
    """
    loss_fn = {
        "edl_digamma": digamma_evd_loss,
        "edl_sumsq": sumsq_evd_loss,
        "edl_nll": nll_evd_loss,
    }
    return loss_fn


def evd_loss_factory(
    name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve an evidential classification loss by name.

    Parameters
    ----------
    name : str
        Registered loss name.

    Returns
    -------
    callable
        Unreduced evidential loss function.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    losses = evd_loss_dict()
    if name not in losses:
        valid = ", ".join(sorted(losses))
        raise ValueError(f"Unknown evidential loss `{name}`. Choose from {valid}.")
    return losses[name]


class EVDLoss(torch.nn.Module):
    """Compute annealed evidential classification loss [1]_.

    References
    ----------
    .. [1] Sensoy et al., "Evidential Deep Learning to Quantify Classification
       Uncertainty," 2018. https://arxiv.org/abs/1806.01768
    """

    def __init__(
        self,
        evd_loss_name: str,
        reduction: Reduction = "none",
        annealing_steps: int = 50000,
        one_hot: bool = True,
        num_classes: int = 5,
        mode: str = "concentration",
    ) -> None:
        """Initialize the evidential classification objective.

        Parameters
        ----------
        evd_loss_name : str
            Registered Bayes-risk loss name.
        reduction : {"none", "mean", "sum"}, default "none"
            Reduction applied across examples.
        annealing_steps : int, default 50000
            Number of steps over which to anneal the KL contribution.
        one_hot : bool, default True
            Convert integer labels to one-hot targets.
        num_classes : int, default 5
            Number of target classes.
        mode : {"concentration", "evidence"}, default "concentration"
            Whether inputs already contain concentrations or nonnegative
            evidence that must be shifted by one.

        Raises
        ------
        ValueError
            If an annealing or class count is invalid, or a mode/reduction is
            unknown.
        """
        super().__init__()
        if annealing_steps < 1:
            raise ValueError(
                "`annealing_steps` must be positive, got " f"{annealing_steps}."
            )
        if num_classes < 2:
            raise ValueError("Evidential classification requires two classes.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unknown reduction method `{reduction}`.")
        if mode not in {"concentration", "evidence"}:
            raise ValueError(f"Unknown evidential input mode `{mode}`.")

        self.annealing_steps = annealing_steps
        self.evd_loss_name = evd_loss_name
        self.evidential_loss = evd_loss_factory(evd_loss_name)
        self.divergence_loss = evd_kl_divergence
        self.reduction: Reduction = reduction
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.mode = mode

    def forward(
        self,
        alpha: torch.Tensor,
        labels: torch.Tensor,
        iteration: int = 0,
    ) -> torch.Tensor:
        """Evaluate the evidential classification objective.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(N, C)`` concentration parameters or evidence.
        labels : torch.Tensor
            ``(N,)`` class IDs when ``one_hot`` is true, otherwise ``(N, C)``
            target probabilities.
        iteration : int, default 0
            Current annealing step.

        Returns
        -------
        torch.Tensor
            Reduced or per-example loss.
        """
        if self.one_hot:
            identity_matrix = torch.eye(
                self.num_classes,
                dtype=alpha.dtype,
                device=alpha.device,
            )
            targets = identity_matrix[labels.long()]
        else:
            targets = labels

        if self.mode != "concentration":
            evidence = alpha
            alpha = evidence + 1.0

        annealing = min(
            1.0,
            max(0.0, float(iteration) / self.annealing_steps),
        )

        evidence_loss = self.evidential_loss(alpha, targets)
        adjusted_concentration = targets + (1 - targets) * alpha
        divergence_loss = self.divergence_loss(adjusted_concentration)

        return _reduce_loss(
            evidence_loss + annealing * divergence_loss,
            self.reduction,
        )

    def __str__(self) -> str:
        return (
            f"EVDLoss(name={self.evd_loss_name}, reduction={self.reduction}, "
            f"one_hot={self.one_hot}, num_classes={self.num_classes}, "
            f"mode={self.mode})"
        )


def nll_regression_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute normal-inverse-gamma negative log-likelihood.

    Parameters
    ----------
    logits : torch.Tensor
        ``(N, 4)`` normal-inverse-gamma parameters in ``gamma``, ``nu``,
        ``alpha``, ``beta`` order.
    targets : torch.Tensor
        ``(N,)`` regression targets.
    eps : float, default 1e-6
        Positive numerical-stability term.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced negative log-likelihood.
    """
    gamma, nu, alpha, beta = logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3]
    scale = 2.0 * beta * (1.0 + nu)
    negative_log_likelihood = (
        0.5 * (log(pi) - torch.log(nu + eps))
        - alpha * torch.log(scale)
        + (alpha + 0.5) * torch.log(nu * (targets - gamma) ** 2 + scale)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return torch.clamp(negative_log_likelihood, min=0.0)


def kld_regression_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Regularize evidence in proportion to absolute prediction error.

    Parameters
    ----------
    logits : torch.Tensor
        ``(N, 4)`` normal-inverse-gamma parameters.
    targets : torch.Tensor
        ``(N,)`` regression targets.
    eps : float, default 1e-6
        Numerical-stability offset applied to the error.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced regularization loss.
    """
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]
    loss = (torch.abs(targets - gamma) + eps) * (2.0 * nu + alpha)
    return loss


def kld_evd_l2_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Regularize evidence in proportion to squared prediction error.

    Parameters
    ----------
    logits : torch.Tensor
        ``(N, 4)`` normal-inverse-gamma parameters.
    targets : torch.Tensor
        ``(N,)`` regression targets.
    eps : float, default 1e-6
        Numerical-stability offset applied to the error.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced regularization loss.
    """
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]
    loss = ((targets - gamma).square() + eps) * (2.0 * nu + alpha)
    return loss


def kl_nig(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 0.01,
) -> torch.Tensor:
    """Compute the error-scaled normal-inverse-gamma KL regularizer.

    Parameters
    ----------
    logits : torch.Tensor
        ``(N, 4)`` normal-inverse-gamma parameters.
    targets : torch.Tensor
        ``(N,)`` regression targets.
    eps : float, default 0.01
        Reference prior offset.

    Returns
    -------
    torch.Tensor
        ``(N,)`` unreduced regularization loss.
    """
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]

    error = torch.abs(targets - gamma) + 1e-6

    divergence = (
        0.5 * (1.0 + eps + 0.001) / (nu + 0.001)
        - 0.5
        - torch.lgamma(alpha / (1.0 + eps))
        + (alpha - (1.0 + eps)) * torch.digamma(alpha)
    )

    loss = divergence * error
    return loss


class EDLRegressionLoss(torch.nn.Module):
    """Combine evidential regression likelihood and evidence regularization."""

    def __init__(
        self,
        reduction: Reduction = "none",
        regularization_weight: float = 0.0,
        kl_mode: str = "evd",
        eps: float = 1e-6,
        annealing_steps: int = 50000,
        logspace: bool = False,
    ) -> None:
        """Initialize the evidential regression objective.

        Parameters
        ----------
        reduction : {"none", "mean", "sum"}, default "none"
            Reduction applied across examples.
        regularization_weight : float, default 0
            Fixed regularization weight when no iteration is supplied.
        kl_mode : {"evd", "kl", "evd_l2"}, default "evd"
            Evidence regularization formulation.
        eps : float, default 1e-6
            Positive numerical-stability term.
        annealing_steps : int, default 50000
            Number of iterations over which to anneal regularization.
        logspace : bool, default False
            Compare the predicted mean with the logarithm of the target.

        Raises
        ------
        ValueError
            If a reduction, regularizer, weight or stability parameter is
            invalid.
        """
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unknown reduction method `{reduction}`.")
        if regularization_weight < 0.0:
            raise ValueError(
                "`regularization_weight` must be nonnegative, got "
                f"{regularization_weight}."
            )
        if eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps}.")
        if annealing_steps < 1:
            raise ValueError(
                "`annealing_steps` must be positive, got " f"{annealing_steps}."
            )
        self.reduction: Reduction = reduction
        self.eps = eps
        self.negative_log_likelihood = nll_regression_loss
        self.kl_mode = kl_mode
        if self.kl_mode == "evd":
            self.regularization_loss = kld_regression_loss
        elif self.kl_mode == "kl":
            self.regularization_loss = kl_nig
        elif self.kl_mode == "evd_l2":
            self.regularization_loss = kld_evd_l2_loss
        else:
            raise ValueError("Unrecognized KL Divergence Error Loss")
        self.regularization_weight = regularization_weight
        self.annealing_steps = annealing_steps
        self.logspace = logspace

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate evidential regression loss.

        Parameters
        ----------
        logits : torch.Tensor
            Normal-inverse-gamma parameters with four values per example.
        targets : torch.Tensor
            ``(N,)`` regression targets.
        iteration : int, optional
            Current annealing iteration. When omitted, use the fixed weight
            ``regularization_weight``.

        Returns
        -------
        torch.Tensor
            Reduced or per-example combined loss.
        torch.Tensor
            Reduced or per-example negative log-likelihood component.

        Raises
        ------
        ValueError
            If targets are not one-dimensional.
        """

        logits = logits.view(-1, 4)
        if self.logspace:
            transformed_targets = torch.log(targets + 1e-6)
        else:
            transformed_targets = targets

        if len(targets.shape) != 1:
            raise ValueError("Expected `len(targets.shape) == 1`.")

        if iteration is not None:
            annealing = min(
                1.0,
                max(
                    0.0,
                    float(iteration) / self.annealing_steps,
                ),
            )
        else:
            annealing = self.regularization_weight

        negative_log_likelihood = self.negative_log_likelihood(
            logits,
            transformed_targets,
            eps=self.eps,
        )
        divergence_loss = self.regularization_loss(
            logits,
            transformed_targets,
            eps=self.eps,
        )

        total_loss = negative_log_likelihood + annealing * divergence_loss
        return (
            _reduce_loss(total_loss, self.reduction),
            _reduce_loss(negative_log_likelihood, self.reduction),
        )
