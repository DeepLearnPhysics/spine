"""Reusable parameter grouping and post-backward gradient diagnostics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from spine.utils.conditional import torch

__all__ = ["GradientTracker", "ParameterGroup"]


@dataclass(frozen=True)
class ParameterGroup:
    """A stable, named selection of trainable parameters.

    Parameters are retained by reference so the same grouping can inspect
    ordinary ``.grad`` tensors or serve as the target of future task-specific
    ``autograd.grad`` calls.

    Attributes
    ----------
    name : str
        Logging-safe group name.
    named_parameters : tuple
        Canonically named trainable parameters in deterministic order.
    """

    name: str
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...]

    @property
    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the parameter references without their canonical names."""
        return tuple(parameter for _, parameter in self.named_parameters)


class GradientTracker:
    """Collect compact gradient statistics after backpropagation.

    The tracker is intentionally passive: it reads gradients already stored
    on parameters and never installs hooks, retains an autograd graph or
    changes an optimizer update. Parameters may be divided into user-defined
    groups with shell-style name patterns. An optional ``global`` group spans
    every registered trainable parameter.

    Notes
    -----
    Statistics are collected after ``backward`` and before ``optimizer.step``.
    Under ``DistributedDataParallel`` this observes the synchronized gradients.
    Missing-gradient fractions count parameter tensors, while RMS uses the
    number of elements represented by gradients that are present. Implicit
    zeros in sparse gradients are included in the RMS denominator.
    """

    _GROUP_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    def __init__(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
        config: bool | Mapping[str, Any] = True,
    ) -> None:
        """Configure cadence and resolve parameter groups.

        Parameters
        ----------
        named_parameters : mapping of str to torch.nn.Parameter
            Canonically named parameters. Only trainable parameters are
            retained, and aliased parameter objects are counted once.
        config : bool or mapping, default True
            ``True`` enables global tracking with an interval of one update.
            A mapping accepts ``interval``, ``include_global`` and ``groups``.
            Group values are one glob pattern or a sequence of glob patterns.

        Raises
        ------
        TypeError
            If configuration or group patterns have the wrong type.
        ValueError
            If cadence, names, patterns or the resulting selections are invalid.
        """
        if not isinstance(config, (bool, Mapping)):
            raise TypeError(
                "Gradient-tracking configuration must be a boolean or mapping."
            )
        if config is False:
            raise ValueError(
                "Construct a gradient tracker only when tracking is enabled."
            )

        settings = {} if config is True else dict(config)
        interval = settings.pop("interval", 1)
        include_global = settings.pop("include_global", True)
        group_config = settings.pop("groups", {})
        if settings:
            unexpected = ", ".join(sorted(settings))
            raise TypeError(f"Unexpected gradient-tracking options: {unexpected}.")
        if not isinstance(interval, int) or isinstance(interval, bool):
            raise TypeError("Gradient-tracking `interval` must be an integer.")
        if interval <= 0:
            raise ValueError("Gradient-tracking `interval` must be positive.")
        if not isinstance(include_global, bool):
            raise TypeError("Gradient-tracking `include_global` must be a boolean.")
        if not isinstance(group_config, Mapping):
            raise TypeError("Gradient-tracking `groups` must be a mapping.")

        self.interval = interval
        self.named_parameters = self._deduplicate_parameters(named_parameters)
        if not self.named_parameters:
            raise ValueError("Gradient tracking requires trainable parameters.")

        groups: dict[str, ParameterGroup] = {}
        if include_global:
            groups["global"] = ParameterGroup(
                "global", tuple(self.named_parameters.items())
            )

        for name, configured_patterns in group_config.items():
            if not isinstance(name, str) or not self._GROUP_NAME.fullmatch(name):
                raise ValueError(
                    "Gradient group names must start with a letter and contain "
                    "only letters, numbers and underscores."
                )
            if name == "global":
                raise ValueError(
                    "`global` is reserved for the complete parameter group."
                )
            patterns = self._normalize_patterns(name, configured_patterns)
            selected = tuple(
                (parameter_name, parameter)
                for parameter_name, parameter in self.named_parameters.items()
                if any(fnmatchcase(parameter_name, pattern) for pattern in patterns)
            )
            if not selected:
                joined = ", ".join(patterns)
                raise ValueError(
                    f"Gradient group `{name}` matched no trainable parameters: {joined}."
                )
            groups[name] = ParameterGroup(name, selected)

        if not groups:
            raise ValueError(
                "Gradient tracking requires `include_global: true` or at least "
                "one configured group."
            )
        self.groups = groups

    @staticmethod
    def from_modules(
        modules: Mapping[str, torch.nn.Module | None],
        config: bool | Mapping[str, Any] = True,
    ) -> "GradientTracker":
        """Build canonical parameter names from one or more modules.

        Module keys become the first component of each parameter name. The
        manager uses ``network`` and ``loss`` so configuration stays stable
        whether or not either module is subsequently wrapped by DDP.

        Parameters
        ----------
        modules : mapping of str to torch.nn.Module or None
            Module namespaces and instances. ``None`` values are skipped.
        config : bool or mapping, default True
            Gradient-tracking configuration forwarded to the constructor.

        Returns
        -------
        GradientTracker
            Tracker containing every unique trainable parameter in the modules.

        Raises
        ------
        TypeError
            If the registry is not a mapping or a value is not a module.
        ValueError
            If a module namespace is empty or no trainable parameters remain.
        """
        if not isinstance(modules, Mapping):
            raise TypeError("Gradient modules must be provided as a mapping.")

        named_parameters: dict[str, torch.nn.Parameter] = {}
        for module_name, module in modules.items():
            if not isinstance(module_name, str) or not module_name:
                raise ValueError("Gradient module names must be non-empty strings.")
            if module is None:
                continue
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Gradient source `{module_name}` must be a module.")
            for parameter_name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    name = f"{module_name}.{parameter_name}"
                    named_parameters[name] = parameter

        return GradientTracker(named_parameters, config)

    @staticmethod
    def _deduplicate_parameters(
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> dict[str, torch.nn.Parameter]:
        """Validate a parameter registry and remove object aliases."""
        if not isinstance(named_parameters, Mapping):
            raise TypeError("Named parameters must be provided as a mapping.")

        unique: dict[str, torch.nn.Parameter] = {}
        seen: set[int] = set()
        for name, parameter in named_parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings.")
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"Gradient source `{name}` is not a parameter.")
            if parameter.requires_grad and id(parameter) not in seen:
                unique[name] = parameter
                seen.add(id(parameter))
        return unique

    @staticmethod
    def _normalize_patterns(name: str, patterns: Any) -> tuple[str, ...]:
        """Normalize one group's glob configuration to a validated tuple."""
        if isinstance(patterns, str):
            patterns = (patterns,)
        elif isinstance(patterns, Sequence):
            patterns = tuple(patterns)
        else:
            raise TypeError(
                f"Gradient group `{name}` must contain a pattern or sequence."
            )
        if not patterns or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise ValueError(
                f"Gradient group `{name}` patterns must be non-empty strings."
            )
        return patterns

    def collect(self, iteration: int) -> dict[str, float | int]:
        """Collect configured gradient metrics for one optimizer iteration.

        Parameters
        ----------
        iteration : int
            Zero-based global optimizer iteration. A cadence of ``N`` records
            completed updates ``N, 2N, ...`` (iterations ``N-1, 2N-1, ...``).

        Returns
        -------
        dict
            Flat scalar metrics suitable for SPINE's CSV and TensorBoard
            logging backends. Off cadence, the same schema is returned with
            ``gradient_sampled=0`` and undefined statistics represented by
            ``NaN`` so fixed-column CSV logs remain valid.

        Raises
        ------
        TypeError
            If ``iteration`` is not an integer.
        ValueError
            If ``iteration`` is negative.
        """
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise TypeError("Gradient-tracking iteration must be an integer.")
        if iteration < 0:
            raise ValueError("Gradient-tracking iteration must be nonnegative.")
        sampled = (iteration + 1) % self.interval == 0
        metrics: dict[str, float | int] = {"gradient_sampled": int(sampled)}
        if not sampled:
            for name in self.groups:
                prefix = f"gradient_{name}"
                for key in (
                    "norm",
                    "rms",
                    "abs_max",
                    "missing_fraction",
                    "nonfinite_count",
                ):
                    metrics[f"{prefix}_{key}"] = math.nan
            return metrics

        with torch.no_grad():
            for name, group in self.groups.items():
                prefix = f"gradient_{name}"
                statistics = self._statistics(group)
                metrics.update(
                    {f"{prefix}_{key}": value for key, value in statistics.items()}
                )
        return metrics

    @staticmethod
    def _statistics(group: ParameterGroup) -> dict[str, float | int]:
        """Reduce one parameter group's current gradients to scalar summaries."""
        sum_squares: torch.Tensor | None = None
        absolute_max: torch.Tensor | None = None
        nonfinite_count: torch.Tensor | None = None
        present_elements = 0
        missing_parameters = 0

        for _, parameter in group.named_parameters:
            gradient = parameter.grad
            if gradient is None:
                missing_parameters += 1
                continue

            # Sparse optimizers store only nonzero values, but RMS describes
            # the full logical parameter tensor, including its implicit zeros.
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            present_elements += gradient.numel()
            squares = values.detach().float().square().sum()
            current_max = (
                values.detach().abs().float().max()
                if values.numel()
                else squares.new_zeros(())
            )
            current_nonfinite = (~torch.isfinite(values)).sum()
            if sum_squares is None:
                sum_squares = squares
                absolute_max = current_max
                nonfinite_count = current_nonfinite
            else:
                sum_squares = sum_squares + squares.to(sum_squares.device)
                absolute_max = torch.maximum(
                    absolute_max, current_max.to(absolute_max.device)
                )
                nonfinite_count = nonfinite_count + current_nonfinite.to(
                    nonfinite_count.device
                )

        parameter_count = len(group.named_parameters)
        missing_fraction = missing_parameters / parameter_count
        if sum_squares is None:
            norm = 0.0
            rms = 0.0
            maximum = 0.0
            nonfinite = 0
        else:
            # Transfer one compact summary per group, avoiding a device
            # synchronization for every individual parameter tensor.
            summary = torch.stack(
                (sum_squares, absolute_max, nonfinite_count.to(sum_squares.dtype))
            ).tolist()
            squared, maximum, nonfinite_value = summary
            norm = math.sqrt(squared)
            rms = math.sqrt(squared / present_elements)
            nonfinite = int(nonfinite_value)

        return {
            "norm": norm,
            "rms": rms,
            "abs_max": maximum,
            "missing_fraction": missing_fraction,
            "nonfinite_count": nonfinite,
        }
