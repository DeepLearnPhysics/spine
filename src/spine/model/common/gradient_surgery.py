"""Task-gradient projection strategies for multi-objective training."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from spine.utils.conditional import torch

from .gradient import GradientTracker, ParameterGroup

if TYPE_CHECKING:  # pragma: no cover - imported only by static type checkers
    from .loss_balancing import LossTerm

__all__ = ["PCGrad"]


class PCGrad:
    """Project conflicting task gradients on shared network parameters.

    PCGrad computes each active objective's gradient independently. For every
    task, the other task gradients are visited in a deterministic pseudo-random
    order. A negative inner product removes the conflicting projection before
    the resulting task gradients are summed. Candidate parameters reached by
    fewer than two active objectives retain their ordinary aggregate gradient.

    The implementation follows Algorithm 1 of Yu et al., *Gradient Surgery for
    Multi-Task Learning* (NeurIPS 2020). It is stateless with respect to model
    checkpoints: the configured seed and global iteration fully determine each
    projection order.

    Notes
    -----
    This class intentionally operates only on network parameters. Loss-module
    parameters, including uncertainty weights, keep the gradient of the
    already-combined loss. Distributed execution is rejected by the manager
    until task gradients can be reduced consistently across ranks.
    """

    name = "pcgrad"

    def __init__(self, parameters: ParameterGroup, seed: int = 0) -> None:
        """Initialize PCGrad over one candidate parameter group.

        Parameters
        ----------
        parameters : ParameterGroup
            Candidate network parameters. Surgery is applied dynamically only
            where at least two active objectives produce gradients.
        seed : int, default 0
            Base seed used to derive per-iteration task projection orders.

        Raises
        ------
        TypeError
            If the parameter group or seed has the wrong type.
        ValueError
            If the parameter group is empty.
        """
        if not isinstance(parameters, ParameterGroup):
            raise TypeError("PCGrad parameters must be a ParameterGroup.")
        if not parameters.named_parameters:
            raise ValueError("PCGrad requires at least one candidate parameter.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("PCGrad `seed` must be an integer.")

        self.parameter_group = parameters
        self.seed = seed

    @classmethod
    def from_module(
        cls,
        network: torch.nn.Module,
        config: Mapping[str, Any],
    ) -> "PCGrad":
        """Construct PCGrad from a network and training configuration.

        Parameters
        ----------
        network : torch.nn.Module
            Unwrapped network whose canonical names receive a ``network.``
            prefix.
        config : mapping
            Configuration with ``name: pcgrad`` and optional ``parameters``
            glob pattern(s) and integer ``seed``. The default candidate pattern
            is ``network.*``.

        Returns
        -------
        PCGrad
            Configured gradient-surgery strategy.

        Raises
        ------
        TypeError
            If the configuration or patterns are malformed.
        ValueError
            If the strategy name or selected parameter group is invalid.
        """
        if not isinstance(config, Mapping):
            raise TypeError("Gradient-balancing configuration must be a mapping.")
        settings = dict(config)
        name = settings.pop("name", None)
        if name is None:
            raise ValueError("Gradient balancing requires a strategy `name`.")
        if not isinstance(name, str) or name.lower() != cls.name:
            raise ValueError("Gradient balancing currently supports only `pcgrad`.")

        patterns = settings.pop("parameters", "network.*")
        seed = settings.pop("seed", 0)
        if settings:
            unexpected = ", ".join(sorted(settings))
            raise TypeError(f"Unexpected gradient-balancing options: {unexpected}.")
        if not isinstance(patterns, (str, Sequence)):
            raise TypeError(
                "PCGrad `parameters` must be a glob pattern or sequence of patterns."
            )

        # Reuse the canonical registry and strict glob validation from passive
        # gradient tracking rather than maintaining a second naming system.
        registry = GradientTracker.from_modules(
            {"network": network},
            {
                "include_global": False,
                "groups": {"pcgrad": patterns},
            },
        )
        return cls(registry.groups["pcgrad"], seed=seed)

    def backward(
        self,
        loss: torch.Tensor,
        terms: Mapping[str, "LossTerm"],
        iteration: int,
    ) -> dict[str, float | int]:
        """Backpropagate once and replace conflicting shared gradients.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar combined objective. Its ordinary backward pass supplies
            gradients outside the dynamically shared parameter subset.
        terms : mapping of str to LossTerm
            Per-objective differentiable contributions after any configured
            fixed or uncertainty weighting.
        iteration : int
            Zero-based global optimizer iteration used for deterministic task
            ordering.

        Returns
        -------
        dict
            Scalar task-gradient and conflict diagnostics.

        Raises
        ------
        TypeError
            If the objective collection, loss values or iteration are invalid.
        ValueError
            If no objective terms are provided or scales are invalid.
        """
        self._validate_inputs(loss, terms, iteration)
        parameters = self.parameter_group.parameters
        active_terms = [
            (name, term)
            for name, term in terms.items()
            if bool(term.active) and float(term.scale) > 0.0
        ]

        task_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for name, term in active_terms:
            objective = term.value * float(term.scale)
            if objective.requires_grad:
                gradients = torch.autograd.grad(
                    objective,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(parameters)
            detached = []
            for gradient in gradients:
                if gradient is not None and gradient.layout != torch.strided:
                    raise TypeError(
                        "PCGrad requires dense gradients on selected parameters."
                    )
                detached.append(None if gradient is None else gradient.detach())
            task_gradients[name] = tuple(detached)

        shared = self._shared_parameter_mask(task_gradients, len(parameters))
        diagnostics = self._diagnostics(terms, task_gradients, shared)
        projected, projection_count = self._project(
            task_gradients,
            shared,
            iteration,
        )

        # Preserve the normal objective for unshared network parameters and
        # all trainable loss-module state, then replace only shared gradients.
        loss.backward()
        if projected:
            for index, parameter in enumerate(parameters):
                if not shared[index]:
                    continue
                combined = sum(
                    (gradients[index] for gradients in projected.values()),
                    start=torch.zeros_like(parameter),
                )
                parameter.grad = combined

        diagnostics["pcgrad_projection_count"] = projection_count
        return diagnostics

    @staticmethod
    def _validate_inputs(
        loss: torch.Tensor,
        terms: Mapping[str, "LossTerm"],
        iteration: int,
    ) -> None:
        """Validate runtime objective metadata before traversing autograd."""
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError("PCGrad requires a scalar loss tensor.")
        if not isinstance(terms, Mapping):
            raise TypeError("PCGrad objective terms must be a mapping.")
        if not terms:
            raise ValueError("PCGrad requires at least one objective term.")
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise TypeError("PCGrad iteration must be an integer.")
        if iteration < 0:
            raise ValueError("PCGrad iteration must be nonnegative.")

        for name, term in terms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("PCGrad objective names must be non-empty strings.")
            value = getattr(term, "value", None)
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"PCGrad objective `{name}` must be a scalar tensor.")
            scale = getattr(term, "scale", None)
            if not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError(f"PCGrad objective `{name}` has an invalid scale.")
            if scale < 0.0:
                raise ValueError(
                    f"PCGrad objective `{name}` scale must be nonnegative."
                )
            if not hasattr(term, "active"):
                raise TypeError(f"PCGrad objective `{name}` has no activity flag.")

    @staticmethod
    def _shared_parameter_mask(
        task_gradients: Mapping[str, tuple[torch.Tensor | None, ...]],
        count: int,
    ) -> tuple[bool, ...]:
        """Identify candidate parameters reached by multiple active tasks."""
        return tuple(
            sum(gradients[index] is not None for gradients in task_gradients.values())
            >= 2
            for index in range(count)
        )

    def _project(
        self,
        task_gradients: Mapping[str, tuple[torch.Tensor | None, ...]],
        shared: tuple[bool, ...],
        iteration: int,
    ) -> tuple[dict[str, list[torch.Tensor]], int]:
        """Apply the paper's randomized pairwise projection rule."""
        if len(task_gradients) < 2 or not any(shared):
            return {}, 0

        names = list(task_gradients)
        projected: dict[str, list[torch.Tensor]] = {}
        for name, gradients in task_gradients.items():
            projected[name] = [
                (torch.zeros_like(parameter) if gradient is None else gradient.clone())
                for parameter, gradient in zip(
                    self.parameter_group.parameters, gradients
                )
            ]

        projection_count = 0
        for task_index, name in enumerate(names):
            others = [other for other in names if other != name]
            # Separate deterministic streams prevent one task's order from
            # depending on the number of random draws used by another task.
            rng = random.Random(self.seed + 1_000_003 * iteration + task_index)
            rng.shuffle(others)
            for other in others:
                dot = self._dot(projected[name], task_gradients[other], shared)
                denominator = self._dot(
                    task_gradients[other], task_gradients[other], shared
                )
                if denominator is None or float(denominator) <= 0.0:
                    continue
                if dot is not None and float(dot) < 0.0:
                    coefficient = dot / denominator
                    for index, other_gradient in enumerate(task_gradients[other]):
                        if shared[index] and other_gradient is not None:
                            projected[name][index] = (
                                projected[name][index] - coefficient * other_gradient
                            )
                    projection_count += 1

        return projected, projection_count

    @staticmethod
    def _dot(
        first: Sequence[torch.Tensor | None],
        second: Sequence[torch.Tensor | None],
        shared: tuple[bool, ...],
    ) -> torch.Tensor | None:
        """Return one inner product without flattening parameter tensors."""
        total: torch.Tensor | None = None
        for enabled, left, right in zip(shared, first, second):
            if not enabled or left is None or right is None:
                continue
            product = torch.sum(left * right)
            total = product if total is None else total + product
        return total

    def _diagnostics(
        self,
        terms: Mapping[str, "LossTerm"],
        task_gradients: Mapping[str, tuple[torch.Tensor | None, ...]],
        shared: tuple[bool, ...],
    ) -> dict[str, float | int]:
        """Summarize original task gradients before projection."""
        metrics: dict[str, float | int] = {
            "pcgrad_active_tasks": len(task_gradients),
            "pcgrad_shared_parameters": sum(shared),
        }
        norms: dict[str, float] = {}
        for name in terms:
            gradients = task_gradients.get(name)
            if gradients is None:
                norm = math.nan
            else:
                squared = self._dot(gradients, gradients, shared)
                norm = 0.0 if squared is None else math.sqrt(float(squared))
            norms[name] = norm
            metrics[f"pcgrad_{name}_gradient_norm"] = norm

        cosines: list[float] = []
        names = list(task_gradients)
        for first_index, first_name in enumerate(names):
            for second_name in names[first_index + 1 :]:
                dot = self._dot(
                    task_gradients[first_name], task_gradients[second_name], shared
                )
                first_norm = norms[first_name]
                second_norm = norms[second_name]
                if dot is not None and first_norm > 0.0 and second_norm > 0.0:
                    cosines.append(float(dot) / (first_norm * second_norm))

        if cosines:
            metrics["pcgrad_mean_cosine"] = sum(cosines) / len(cosines)
            metrics["pcgrad_conflict_fraction"] = sum(
                cosine < 0.0 for cosine in cosines
            ) / len(cosines)
        else:
            metrics["pcgrad_mean_cosine"] = math.nan
            metrics["pcgrad_conflict_fraction"] = math.nan
        return metrics
