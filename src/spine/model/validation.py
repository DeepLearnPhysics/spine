"""Checkpoint-bound validation and early-stopping orchestration."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from numbers import Real
from typing import Any

import numpy as np

from spine.io import IOManager
from spine.logging import LogManager, logger
from spine.utils.conditional import torch
from spine.utils.stopwatch import StopwatchManager
from spine.utils.torch import runtime

from .manager import ModelManager

__all__ = ["ValidationManager"]


def _metric_improved(
    value: float,
    best: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    """Return whether ``value`` improves on a monitored best value."""
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


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
        improved = _metric_improved(value, self.best, self.mode, self.min_delta)

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


class BestCheckpoint:
    """Track which checkpoint produced the best validation metric.

    Parameters
    ----------
    monitor : str
        Validation scalar used to select the best checkpoint.
    mode : {'min', 'max'}
        Direction in which the monitored metric improves.
    min_delta : float
        Minimum absolute change required to replace the best checkpoint.
    path : str or None
        Optional stable checkpoint destination. When omitted, the model weight
        prefix is suffixed with ``-best.ckpt``.
    best : float or None
        Best monitored value observed so far.
    """

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "min",
        min_delta: float = 0.0,
        path: str | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the best-checkpoint selection policy.

        Parameters
        ----------
        monitor : str, default 'loss'
            Validation scalar used to select the best checkpoint.
        mode : {'min', 'max'}, default 'min'
            Direction in which the monitored metric improves.
        min_delta : float, default 0.0
            Minimum absolute change required to replace the best checkpoint.
        path : str, optional
            Stable destination for the promoted checkpoint.
        state : mapping, optional
            Previously checkpointed best-metric progress.

        Raises
        ------
        ValueError
            If the mode or minimum delta is invalid.
        TypeError
            If the destination path is not a string.
        """
        if mode not in {"min", "max"}:
            raise ValueError("Best-checkpoint `mode` must be 'min' or 'max'.")
        if min_delta < 0.0:
            raise ValueError("Best-checkpoint `min_delta` must be non-negative.")
        if path is not None and not isinstance(path, str):
            raise TypeError("Best-checkpoint `path` must be a string.")

        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.path = path
        self.best: float | None = None

        if state is not None:
            self.restore(state)

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore compatible best-metric progress from a checkpoint.

        Parameters
        ----------
        state : mapping
            Serialized best-checkpoint state.

        Raises
        ------
        ValueError
            If the checkpoint monitors a different metric or direction.
        """
        if state.get("monitor") != self.monitor or state.get("mode") != self.mode:
            raise ValueError(
                "Checkpoint best-checkpoint policy does not match the current "
                "`monitor` and `mode`."
            )
        best = state.get("best")
        self.best = None if best is None else float(best)

    def update(self, metrics: Mapping[str, float]) -> bool:
        """Update the best metric and select checkpoint promotion.

        Parameters
        ----------
        metrics : mapping[str, float]
            Globally reduced validation metrics.

        Returns
        -------
        bool
            Whether the current checkpoint should replace the saved best.

        Raises
        ------
        KeyError
            If the monitored metric is absent.
        """
        if self.monitor not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Best-checkpoint metric `{self.monitor}` was not produced. "
                f"Available scalar metrics: {available}."
            )

        value = float(metrics[self.monitor])
        improved = _metric_improved(value, self.best, self.mode, self.min_delta)
        if improved:
            self.best = value
        return improved

    def state_dict(self) -> dict[str, Any]:
        """Return serializable best-checkpoint progress.

        Returns
        -------
        dict
            Policy parameters and best monitored value.
        """
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best": self.best,
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
    best_checkpoint : BestCheckpoint or None
        Optional best-checkpoint selection policy.
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
        log_dir: str | None = None,
        prefix_log: bool = False,
        overwrite_log: bool = False,
        csv_buffer_size: int = 1,
        log_step: int = 1,
    ) -> None:
        """Build a validation loader and optional validation policies.

        ``cfg`` only describes validation sources, the fraction of the loader
        to visit, early stopping and best-checkpoint selection. The training
        loader supplies the dataset schema, batching, collation and worker
        configuration.

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
        log_dir : str, optional
            Directory in which to write checkpoint validation log segments.
            If omitted, validation CSV logging is disabled.
        prefix_log : bool, default False
            Prefix validation log names with the input-derived stem.
        overwrite_log : bool, default False
            Allow an existing validation log segment to be overwritten.
        csv_buffer_size : int, default 1
            Buffer size forwarded to the validation CSV logger.
        log_step : int, default 1
            Number of validation batches between stdout summaries.

        Raises
        ------
        ValueError
            If ``fraction`` is outside ``(0, 1]``.
        TypeError
            If a validation policy has an invalid configuration type.
        """
        # Parse validation-owned scheduling options
        cfg = deepcopy(dict(cfg))
        fraction = cfg.pop("fraction", 1.0)
        early_cfg = cfg.pop("early_stopping", None)
        best_cfg = cfg.pop("best_checkpoint", None)
        if not isinstance(fraction, Real):
            raise TypeError("Validation `fraction` must be a real number.")
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
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
        self.rank = rank
        self.log_dir = log_dir
        self.prefix_log = prefix_log
        self.overwrite_log = overwrite_log
        self.csv_buffer_size = csv_buffer_size
        self.log_step = log_step
        self.main_process = rank is None or rank == 0
        assert self.io.loader is not None
        self.num_iterations = max(1, math.ceil(fraction * len(self.io.loader)))

        # Restore optional validation policies from the loaded checkpoint.
        restored = model.checkpoint_validation or {}
        self.early_stopping = None
        if early_cfg is not None:
            if not isinstance(early_cfg, Mapping):
                raise TypeError("`validation.early_stopping` must be a mapping.")
            state = restored.get("early_stopping")
            self.early_stopping = EarlyStopping(**early_cfg, state=state)

        self.best_checkpoint = None
        if best_cfg is not None and best_cfg is not False:
            if best_cfg is True:
                best_cfg = {}
            elif not isinstance(best_cfg, Mapping):
                raise TypeError(
                    "`validation.best_checkpoint` must be a boolean or mapping."
                )
            state = restored.get("best_checkpoint")
            self.best_checkpoint = BestCheckpoint(**best_cfg, state=state)

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

    def run(self, iteration: int, epoch: float | None = None) -> dict[str, float]:
        """Evaluate and average scalar outputs over validation batches.

        Parameters
        ----------
        iteration : int
            Current training iteration forwarded to time-dependent losses.
        epoch : float, optional
            Current training epoch recorded alongside validation batches.

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
        assert self.io.loader is not None
        sampler = self.io.loader.sampler
        if self.distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        self.io.reset_loader()

        # Each checkpoint validation pass mirrors a standalone inference run
        # in its own log segment. The filename boundary is the next training
        # iteration, matching checkpoint resume and TrainDrawer conventions.
        log_manager = self.initialize_log(iteration)
        watch = StopwatchManager()
        watch.initialize(["iteration", "model"])
        if self.main_process:
            logger.info(
                "Training iteration %d complete; starting validation (%d batch%s).",
                iteration,
                self.num_iterations,
                "" if self.num_iterations == 1 else "es",
            )

        # Accumulate scalar outputs and ignore structured predictions
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        try:
            for val_iteration in range(self.num_iterations):
                watch.start("iteration")
                data = self.io.load()
                watch.start("model")
                result = self.model.evaluate(data, iteration)
                watch.stop("model")
                data.update(result)
                if hasattr(self.io, "watch"):
                    watch.update(self.io.watch)
                watch.stop("iteration")

                if log_manager is not None:
                    log_row = log_manager.append(data, watch, val_iteration, epoch)
                    if ((val_iteration + 1) % self.log_step) == 0:
                        log_manager.log_stdout_summary(
                            log_row,
                            data,
                            watch,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            val_iteration,
                            epoch,
                            model_train=False,
                            rank=self.rank,
                            distributed=self.distributed,
                            main_process=self.main_process,
                            mode="validation",
                            total_iterations=self.num_iterations,
                        )

                for key, value in result.items():
                    scalar = self.to_scalar(value)
                    if scalar is not None:
                        totals[key] = totals.get(key, 0.0) + scalar
                        counts[key] = counts.get(key, 0) + 1
        finally:
            if log_manager is not None:
                log_manager.close()

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

        metrics = {key: total / counts[key] for key, total in totals.items()}
        if self.main_process:
            summary = ", ".join(
                f"{key}={value:.6g}" for key, value in sorted(metrics.items())
            )
            logger.info(
                "Validation complete at training iteration %d: %s.",
                iteration,
                summary,
            )
        return metrics

    def initialize_log(self, iteration: int) -> LogManager | None:
        """Create the inference-style log segment for one validation pass."""
        if getattr(self, "log_dir", None) is None:
            return None
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        suffix = "" if not self.distributed else f"_proc{self.rank}"
        log_name = f"validation{suffix}_log-{iteration + 1:07d}.csv"
        if self.prefix_log:
            log_name = self.io.format_log_name(log_name, self.log_dir)
        return LogManager(
            os.path.join(self.log_dir, log_name),
            overwrite=self.overwrite_log,
            buffer_size=self.csv_buffer_size,
        )

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

    def update_best_checkpoint(self, metrics: Mapping[str, float]) -> bool:
        """Update and select the best checkpoint observed so far.

        Parameters
        ----------
        metrics : mapping[str, float]
            Globally reduced validation metrics.

        Returns
        -------
        bool
            Whether the current checkpoint should replace the saved best.
        """
        if self.best_checkpoint is None:
            return False
        return self.best_checkpoint.update(metrics)

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
        if self.best_checkpoint is not None:
            state["best_checkpoint"] = self.best_checkpoint.state_dict()
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
        if runtime.is_tensor(value) and value.numel() == 1:
            return float(value.detach().item())
        return None

    def close(self) -> None:
        """Close resources owned by the validation input manager."""
        self.io.close()
