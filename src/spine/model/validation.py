"""Checkpoint-bound validation and early-stopping orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np

from spine.io import IOManager
from spine.utils.conditional import torch

from .manager import ModelManager

__all__ = ["ValidationManager"]


class EarlyStopping:
    """Track validation improvement across checkpoint boundaries.

    Attributes
    ----------
    monitor : str
        Name of the scalar validation metric being monitored.
    mode : {'min', 'max'}
        Direction in which the monitored metric improves.
    patience : int
        Number of consecutive non-improving validations tolerated.
    min_delta : float
        Minimum absolute change required to reset the patience counter.
    best : float or None
        Best monitored value observed so far.
    bad_checks : int
        Number of consecutive non-improving validation checks.
    """

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an early-stopping policy.

        Parameters
        ----------
        monitor : str, default 'loss'
            Validation metric to monitor.
        mode : {'min', 'max'}, default 'min'
            Whether smaller or larger monitored values are improvements.
        patience : int, default 5
            Number of consecutive non-improving validations tolerated.
        min_delta : float, default 0.0
            Minimum absolute change required to count as an improvement.
        state : mapping, optional
            Previously checkpointed state to restore.

        Raises
        ------
        ValueError
            If the mode, patience or minimum delta is invalid, or if restored
            state monitors a different metric or direction.
        """
        if mode not in {"min", "max"}:
            raise ValueError("Early-stopping `mode` must be 'min' or 'max'.")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError(
                "Early-stopping `patience` must be a non-negative integer."
            )
        if min_delta < 0.0:
            raise ValueError("Early-stopping `min_delta` must be non-negative.")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_checks = 0

        if state is not None:
            self.restore(state)

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore compatible progress from a checkpoint.

        Parameters
        ----------
        state : mapping
            Serialized early-stopping state from a prior checkpoint.

        Raises
        ------
        ValueError
            If the checkpoint monitors a different metric or direction.
        """
        if state.get("monitor") != self.monitor or state.get("mode") != self.mode:
            raise ValueError(
                "Checkpoint early-stopping policy does not match the current "
                "`monitor` and `mode`."
            )

        best = state.get("best")
        self.best = None if best is None else float(best)
        self.bad_checks = int(state.get("bad_checks", 0))

    def update(self, metrics: Mapping[str, float]) -> bool:
        """Update progress from one completed validation pass.

        Parameters
        ----------
        metrics : mapping[str, float]
            Globally reduced scalar validation metrics.

        Returns
        -------
        bool
            Whether the configured patience has been exhausted.

        Raises
        ------
        KeyError
            If the monitored metric is absent from ``metrics``.
        """
        if self.monitor not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Early-stopping metric `{self.monitor}` was not produced. "
                f"Available scalar metrics: {available}."
            )

        value = float(metrics[self.monitor])
        improved = self.best is None
        if self.best is not None and self.mode == "min":
            improved = value < self.best - self.min_delta
        elif self.best is not None:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_checks = 0
        else:
            self.bad_checks += 1

        return not improved and self.bad_checks >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """Return serializable early-stopping state.

        Returns
        -------
        dict
            Policy parameters, best value and current patience counter.
        """
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_checks": self.bad_checks,
        }


class ValidationManager:
    """Run deterministic validation against the live training model.

    The manager derives its loader from the training loader configuration,
    replacing only dataset sources and stochastic sampling behavior. It owns
    no model: validation reuses the exact in-memory model and DDP wrapper used
    by training.

    Attributes
    ----------
    io : IOManager
        Validation-only input manager.
    model : ModelManager
        Live model manager shared with the training driver.
    distributed : bool
        Whether scalar metrics must be reduced across ranks.
    num_iterations : int
        Number of validation batches processed per checkpoint.
    early_stopping : EarlyStopping or None
        Optional early-stopping policy.
    """

    SOURCE_KEYS = frozenset({"file_keys", "file_list"})
    FILTER_KEYS = frozenset(
        {
            "n_entry",
            "n_skip",
            "entry_list",
            "skip_entry_list",
            "run_event_list",
            "skip_run_event_list",
        }
    )

    def __init__(
        self,
        cfg: Mapping[str, Any],
        loader: Mapping[str, Any],
        model: ModelManager,
        *,
        rank: int | None,
        dtype: str,
        world_size: int,
        distributed: bool,
        seed: int,
    ) -> None:
        """Build a validation loader and optional early-stopping policy.

        ``cfg`` only describes validation sources, the fraction of the loader
        to visit, and early stopping. The training loader supplies the dataset
        schema, batching, collation and worker configuration.

        Parameters
        ----------
        cfg : mapping
            Validation source, fraction and early-stopping configuration.
        loader : mapping
            Training loader configuration used as the derivation template.
        model : ModelManager
            Live training model used for validation forwards.
        rank : int, optional
            Process rank used by a distributed validation loader.
        dtype : str
            Floating-point dtype forwarded to the derived loader.
        world_size : int
            Total number of loader/model processes.
        distributed : bool
            Whether validation data and metrics are distributed across ranks.
        seed : int
            Fixed seed used for deterministic joint overlay selection.

        Raises
        ------
        ValueError
            If ``fraction`` is outside ``(0, 1]``.
        TypeError
            If the early-stopping configuration is not a mapping.
        """
        # Parse validation-owned scheduling options
        cfg = deepcopy(dict(cfg))
        fraction = cfg.pop("fraction", 1.0)
        early_cfg = cfg.pop("early_stopping", None)
        if not isinstance(fraction, Real) or not 0.0 < fraction <= 1.0:
            raise ValueError("Validation `fraction` must be in the interval (0, 1].")

        # Derive and initialize the validation input pipeline
        loader_cfg = self.build_loader_config(loader, cfg, seed)
        self.io = IOManager(
            loader=loader_cfg,
            rank=rank,
            dtype=dtype,
            world_size=world_size,
            distributed=distributed,
        )
        self.model = model
        self.distributed = distributed
        self.num_iterations = max(1, math.ceil(float(fraction) * len(self.io.loader)))

        # Restore optional early-stopping progress from the loaded checkpoint
        self.early_stopping = None
        if early_cfg is not None:
            if not isinstance(early_cfg, Mapping):
                raise TypeError("`validation.early_stopping` must be a mapping.")
            restored = model.checkpoint_validation or {}
            state = restored.get("early_stopping")
            self.early_stopping = EarlyStopping(**early_cfg, state=state)

    @classmethod
    def build_loader_config(
        cls,
        loader: Mapping[str, Any],
        cfg: Mapping[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """Derive a validation loader by replacing dataset source leaves.

        Parameters
        ----------
        loader : mapping
            Training loader configuration to copy.
        cfg : mapping
            Validation source overrides after manager-owned options have been
            removed. This mapping is consumed while it is validated.
        seed : int
            Deterministic seed for joint overlay selection.

        Returns
        -------
        dict
            Independent validation loader configuration.

        Raises
        ------
        TypeError
            If the training dataset or a composite source is not inline.
        ValueError
            If source overrides do not match the dataset topology.
        KeyError
            If unrecognized validation keys remain after source processing.
        """
        # Copy the loader and remove training-only dataset behavior
        loader_cfg = deepcopy(dict(loader))
        cfg = dict(cfg)
        dataset = loader_cfg.get("dataset")
        if not isinstance(dataset, Mapping):
            raise TypeError(
                "On-the-fly validation requires an inline dataset mapping in "
                "`io.loader.dataset`."
            )

        dataset = deepcopy(dict(dataset))
        cls.strip_runtime_options(dataset)
        dataset_name = dataset.get("name")

        # Replace sources according to the dataset topology
        if dataset_name == "joint":
            sources = cls.pop_composite_sources(cfg, {"primary", "secondary"})
            cls.replace_source(dataset, "primary", sources["primary"])
            cls.replace_source(dataset, "secondary", sources["secondary"])

            # Preserve overlay frequency with repeatable pair selection
            pair_probability = 1.0
            train_sampler = loader_cfg.get("sampler")
            if isinstance(train_sampler, Mapping):
                pair_probability = train_sampler.get("pair_probability", 1.0)
            loader_cfg["sampler"] = {
                "name": "joint_sequential",
                "seed": seed,
                "pair_probability": pair_probability,
            }

        elif dataset_name == "mixed":
            sources = cls.pop_composite_sources(cfg, {"larcv", "hdf5"})
            cls.replace_source(dataset, "larcv", sources["larcv"])
            cls.replace_source(dataset, "hdf5", sources["hdf5"])
            loader_cfg.pop("sampler", None)

        else:
            source = cls.pop_simple_source(cfg)
            cls.apply_source(dataset, source)
            loader_cfg.pop("sampler", None)

        # Reject options outside the intentionally narrow validation schema
        if cfg:
            invalid = ", ".join(sorted(cfg))
            raise KeyError(f"Unrecognized keys in `validation`: {invalid}.")

        # Force deterministic loader-level traversal
        loader_cfg["dataset"] = dataset
        loader_cfg.pop("entry_list", None)
        loader_cfg["shuffle"] = False
        return loader_cfg

    @classmethod
    def strip_runtime_options(cls, config: dict[str, Any]) -> None:
        """Remove training-only options recursively from a dataset config.

        Parameters
        ----------
        config : dict
            Dataset or nested source configuration modified in place.
        """
        for key in (*cls.SOURCE_KEYS, *cls.FILTER_KEYS, "augment"):
            config.pop(key, None)
        for value in config.values():
            if isinstance(value, dict):
                cls.strip_runtime_options(value)

    @classmethod
    def pop_simple_source(cls, cfg: dict[str, Any]) -> Mapping[str, Any]:
        """Extract one ordinary validation source from the top-level block.

        Parameters
        ----------
        cfg : dict
            Validation configuration modified in place.

        Returns
        -------
        mapping
            Source mapping containing exactly one file selector.

        Raises
        ------
        ValueError
            If named composite sources are provided.
        """
        if "sources" in cfg:
            raise ValueError(
                "`validation.sources` is only valid for composite datasets."
            )
        source = {key: cfg.pop(key) for key in cls.SOURCE_KEYS if key in cfg}
        cls.validate_source(source, "validation")
        return source

    @classmethod
    def pop_composite_sources(
        cls, cfg: dict[str, Any], expected: set[str]
    ) -> dict[str, Mapping[str, Any]]:
        """Extract and validate named sources for a composite dataset.

        Parameters
        ----------
        cfg : dict
            Validation configuration modified in place.
        expected : set[str]
            Exact set of source names required by the dataset topology.

        Returns
        -------
        dict[str, mapping]
            Validated source mappings keyed by source name.

        Raises
        ------
        ValueError
            If flat selectors are used or the source-name set is incomplete.
        TypeError
            If an individual source block is not a mapping.
        """
        if any(key in cfg for key in cls.SOURCE_KEYS):
            raise ValueError(
                "Composite validation datasets require named `validation.sources`."
            )
        sources = cfg.pop("sources", None)
        if not isinstance(sources, Mapping) or set(sources) != expected:
            names = ", ".join(sorted(expected))
            raise ValueError(f"Validation sources must provide exactly: {names}.")

        result = {}
        for name, source in sources.items():
            if not isinstance(source, Mapping):
                raise TypeError(f"Validation source `{name}` must be a mapping.")
            source = dict(source)
            cls.validate_source(source, f"validation.sources.{name}")
            result[name] = source
        return result

    @classmethod
    def validate_source(cls, source: Mapping[str, Any], label: str) -> None:
        """Require exactly one supported file selector in a source block.

        Parameters
        ----------
        source : mapping
            Candidate source override.
        label : str
            Configuration path used in validation errors.

        Raises
        ------
        KeyError
            If the source contains unsupported keys.
        ValueError
            If it does not provide exactly one file selector.
        """
        invalid = set(source) - cls.SOURCE_KEYS
        if invalid:
            names = ", ".join(sorted(invalid))
            raise KeyError(f"Unrecognized keys in `{label}`: {names}.")
        if len(source) != 1:
            raise ValueError(
                f"`{label}` must provide one of `file_keys` or `file_list`."
            )

    @classmethod
    def replace_source(
        cls,
        dataset: dict[str, Any],
        name: str,
        source: Mapping[str, Any],
    ) -> None:
        """Replace one inline source block of a composite dataset.

        Parameters
        ----------
        dataset : dict
            Composite dataset configuration modified in place.
        name : str
            Source block name within ``dataset``.
        source : mapping
            Validated validation-file selector.

        Raises
        ------
        TypeError
            If the inherited source is not an inline mapping.
        """
        source_cfg = dataset.get(name)
        if not isinstance(source_cfg, Mapping):
            raise TypeError(
                f"Validation requires inline `io.loader.dataset.{name}` configuration."
            )
        source_cfg = dict(source_cfg)
        cls.apply_source(source_cfg, source)
        dataset[name] = source_cfg

    @classmethod
    def apply_source(cls, dataset: dict[str, Any], source: Mapping[str, Any]) -> None:
        """Replace file selection on one resolved dataset source.

        Parameters
        ----------
        dataset : dict
            Dataset source configuration modified in place.
        source : mapping
            Validated validation-file selector.
        """
        for key in (*cls.SOURCE_KEYS, *cls.FILTER_KEYS):
            dataset.pop(key, None)
        dataset.update(source)

    def run(self, iteration: int) -> dict[str, float]:
        """Evaluate and average scalar outputs over validation batches.

        Parameters
        ----------
        iteration : int
            Current training iteration forwarded to time-dependent losses.

        Returns
        -------
        dict[str, float]
            Mean scalar metrics, reduced across ranks when distributed.

        Raises
        ------
        RuntimeError
            If the model does not produce any scalar validation outputs.
        """
        # Reset deterministic sampler and loader state for this pass
        sampler = self.io.loader.sampler
        if self.distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        self.io.reset_loader()

        # Accumulate scalar outputs and ignore structured predictions
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _ in range(self.num_iterations):
            result = self.model.evaluate(self.io.load(), iteration)
            for key, value in result.items():
                scalar = self.to_scalar(value)
                if scalar is not None:
                    totals[key] = totals.get(key, 0.0) + scalar
                    counts[key] = counts.get(key, 0) + 1

        if not totals:
            raise RuntimeError("Validation did not produce any scalar metrics.")

        # Reduce sums and observation counts over the process group
        if self.distributed:
            for key in sorted(totals):
                reduced = torch.tensor(
                    [totals[key], counts[key]],
                    dtype=torch.float64,
                    device=self.model.device,
                )
                torch.distributed.all_reduce(reduced)
                totals[key], counts[key] = reduced.tolist()

        return {key: totals[key] / counts[key] for key in totals}

    def update_early_stopping(self, metrics: Mapping[str, float]) -> bool:
        """Update early stopping after a validation pass.

        Parameters
        ----------
        metrics : mapping[str, float]
            Globally reduced validation metrics.

        Returns
        -------
        bool
            Whether training should stop. Always ``False`` when early stopping
            is disabled.
        """
        if self.early_stopping is None:
            return False
        return self.early_stopping.update(metrics)

    def checkpoint_state(self, metrics: Mapping[str, float]) -> dict[str, Any]:
        """Build validation state to store alongside checkpoint weights.

        Parameters
        ----------
        metrics : mapping[str, float]
            Metrics produced from the checkpoint-bound validation pass.

        Returns
        -------
        dict
            Validation metrics and optional early-stopping progress.
        """
        state: dict[str, Any] = {"metrics": dict(metrics)}
        if self.early_stopping is not None:
            state["early_stopping"] = self.early_stopping.state_dict()
        return state

    @staticmethod
    def to_scalar(value: Any) -> float | None:
        """Convert a scalar numeric output to a Python float.

        Parameters
        ----------
        value : object
            Candidate model or loss output.

        Returns
        -------
        float or None
            Scalar value, or ``None`` for structured/non-scalar outputs.
        """
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, np.ndarray) and value.size == 1:
            return float(value.item())
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().item())
        return None

    def close(self) -> None:
        """Close resources owned by the validation input manager."""
        self.io.close()
