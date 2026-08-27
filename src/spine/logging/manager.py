"""Structured scalar logging manager."""

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import psutil

from spine.utils.torch import runtime

from .csv import CSVLogger
from .logger import logger

__all__ = ["LogManager"]


class LogManager:
    """Manage structured scalar logs for a driver-like processing loop.

    The manager writes one flat scalar row to CSV on every call and can mirror
    numeric entries to TensorBoard. It also owns the human-readable progress
    table printed periodically during training or inference, along with the
    checkpoint and validation lifecycle messages which surround those tables.
    """

    # Column widths shared by all human-readable iteration tables
    STDOUT_WIDTHS: tuple[int, ...] = (20, 20, 9, 9)
    STDOUT_RANK_WIDTH: int = 5

    def __init__(
        self,
        file_name: str,
        overwrite: bool = False,
        buffer_size: int = 1,
        tensorboard: bool | Mapping[str, Any] | None = None,
        tensorboard_dir: str | None = None,
    ) -> None:
        """Initialize scalar logging backends.

        Parameters
        ----------
        file_name : str
            CSV log file path.
        overwrite : bool, default False
            If ``True``, overwrite an existing CSV log file.
        buffer_size : int, default 1
            CSV writer buffer size.
        tensorboard : bool | Mapping[str, Any] | None, optional
            TensorBoard logging configuration. ``False`` or ``None`` disable
            TensorBoard logging, ``True`` uses default settings, and a mapping
            forwards keyword arguments to the TensorBoard writer.
        tensorboard_dir : str | None, optional
            Default TensorBoard event-file directory. If ``tensorboard`` is a
            mapping with a ``log_dir`` key, that value takes precedence.
        """
        self.csv_logger = CSVLogger(
            file_name, overwrite=overwrite, buffer_size=buffer_size
        )
        self.tb_logger = self.initialize_tensorboard_logger(
            tensorboard, tensorboard_dir
        )

    @staticmethod
    def initialize_tensorboard_logger(
        tensorboard: bool | Mapping[str, Any] | None,
        tensorboard_dir: str | None = None,
    ) -> Any | None:
        """Initialize an optional TensorBoard summary writer.

        Parameters
        ----------
        tensorboard : bool | Mapping[str, Any] | None
            TensorBoard logging configuration.
        tensorboard_dir : str | None, optional
            Default TensorBoard event-file directory.

        Returns
        -------
        Any | None
            TensorBoard summary writer instance when enabled, otherwise
            ``None``.
        """
        if not tensorboard:
            return None

        tb_cfg = {} if tensorboard is True else dict(tensorboard)
        tb_dir = tb_cfg.pop("log_dir", None)
        if tb_dir is None:
            tb_dir = tensorboard_dir
        elif not os.path.isabs(tb_dir) and tensorboard_dir is not None:
            tb_dir = os.path.join(os.path.dirname(tensorboard_dir), tb_dir)

        if tb_dir is None:
            raise ValueError(
                "A TensorBoard log directory is required when TensorBoard "
                "logging is enabled."
            )

        return runtime.create_summary_writer(tb_dir, **tb_cfg)

    def append(
        self,
        data: Mapping[str, Any],
        watch: Any,
        iteration: int,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Collect and write one scalar log row.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data products returned by the processing loop.
        watch : object
            Stopwatch manager with ``items`` and ``time`` methods.
        iteration : int
            Iteration counter.
        epoch : float | None, optional
            Progress in the training loop measured in epochs.

        Returns
        -------
        dict[str, Any]
            Flat log row written to all enabled structured backends.
        """
        log_row = self.collect(data, watch, iteration, epoch)
        self.csv_logger.append(log_row)
        self.append_tensorboard(log_row, iteration)
        return log_row

    def collect(
        self,
        data: Mapping[str, Any],
        watch: Any,
        iteration: int,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Collect scalar iteration metrics into one flat log row.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data products returned by the processing loop.
        watch : object
            Stopwatch manager with ``items`` and ``time`` methods.
        iteration : int
            Iteration counter.
        epoch : float | None, optional
            Progress in the training loop measured in epochs.

        Returns
        -------
        dict[str, Any]
            Flat row of scalar values ready to be written to logging backends.
        """
        first_entry = get_first_entry(data["index"])
        log_row = {"iter": iteration, "epoch": epoch, "first_entry": first_entry}
        log_row.update(self.get_memory_metrics())
        log_row.update(self.get_watch_metrics(watch))

        for key, value in data.items():
            if np.isscalar(value):
                log_row[key] = value
            elif runtime.is_tensor(value) and value.dim() == 0:
                log_row[key] = value.item()

        return log_row

    @staticmethod
    def get_memory_metrics() -> dict[str, float]:
        """Collect CPU and GPU memory metrics for the current process."""
        metrics = {
            "cpu_mem": psutil.virtual_memory().used / 1.0e9,
            "cpu_mem_perc": psutil.virtual_memory().percent,
            "gpu_mem": 0.0,
            "gpu_mem_perc": 0.0,
        }
        if runtime.cuda_is_available():
            gpu_total = runtime.cuda_mem_info()[-1] / 1.0e9
            metrics["gpu_mem"] = runtime.cuda_max_memory_allocated() / 1.0e9
            metrics["gpu_mem_perc"] = 100 * metrics["gpu_mem"] / gpu_total

        return metrics

    @staticmethod
    def get_watch_metrics(watch: Any) -> dict[str, float]:
        """Flatten stopwatch timings into loggable scalar metrics."""
        metrics: dict[str, float] = {}
        suffix = "_time"
        for key, timer in watch.items():
            time_iter, time_sum = timer.time, timer.time_sum
            metrics[f"{key}{suffix}"] = time_iter.wall
            metrics[f"{key}{suffix}_cpu"] = time_iter.cpu
            metrics[f"{key}{suffix}_sum"] = time_sum.wall
            metrics[f"{key}{suffix}_sum_cpu"] = time_sum.cpu

        return metrics

    def append_tensorboard(self, log_row: Mapping[str, Any], iteration: int) -> None:
        """Write collected scalar metrics to TensorBoard, if enabled."""
        if self.tb_logger is None:
            return

        for key, value in log_row.items():
            if key == "iter":
                continue
            if isinstance(value, bool):
                self.tb_logger.add_scalar(key, int(value), iteration)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                self.tb_logger.add_scalar(key, float(value), iteration)

    @staticmethod
    def stdout_table_width(distributed: bool = False) -> int:
        """Return the rendered width of an iteration-summary table.

        Parameters
        ----------
        distributed : bool, default False
            Whether the table includes the leading process-rank column.

        Returns
        -------
        int
            Number of characters occupied by one complete table row.
        """
        widths = list(LogManager.STDOUT_WIDTHS)
        if distributed:
            widths.insert(0, LogManager.STDOUT_RANK_WIDTH)

        # Account for the left margin and each column delimiter
        return 4 + sum(widths) + 2 * (len(widths) - 1) + 1

    @staticmethod
    def log_checkpoint_start(
        iteration: int,
        epoch: float,
        validation_batches: int | None,
        distributed: bool = False,
    ) -> None:
        """Open a human-readable checkpoint progress section.

        The section remains open while optional validation and checkpoint
        serialization run. :meth:`log_checkpoint_complete` closes it only
        after the checkpoint has been persisted successfully.

        Parameters
        ----------
        iteration : int
            Training iteration associated with the checkpoint.
        epoch : float
            Training progress, expressed in epochs.
        validation_batches : int, optional
            Number of validation batches scheduled at this boundary. ``None``
            indicates that on-the-fly validation is disabled.
        distributed : bool, default False
            Whether progress tables include a process-rank column.
        """
        # Match the section boundary to the associated progress-table width
        separator = "=" * LogManager.stdout_table_width(distributed)
        validation_label = (
            "disabled" if validation_batches is None else str(validation_batches)
        )

        logger.info(
            "%s\n"
            "CHECKPOINT\n"
            "Training iteration: %d\n"
            "Epoch:              %.3f\n"
            "Validation batches: %s\n",
            separator,
            iteration,
            epoch,
            validation_label,
        )

    @staticmethod
    def log_validation_start() -> None:
        """Mark the start of validation within an open checkpoint section."""
        logger.info("VALIDATION START\n")

    @staticmethod
    def log_validation_complete(metrics: Mapping[str, float]) -> None:
        """Render aggregate validation metrics inside a checkpoint section.

        Parameters
        ----------
        metrics : mapping[str, float]
            Globally reduced scalar validation metrics.
        """
        # Align metric values while retaining deterministic alphabetical order
        key_width = max(map(len, metrics))
        summary = "\n".join(
            f"  {key:<{key_width}}: {value:.6g}"
            for key, value in sorted(metrics.items())
        )

        logger.info("VALIDATION COMPLETE\nMetrics:\n%s\n", summary)

    @staticmethod
    def log_checkpoint_saving() -> None:
        """Mark serialization within an open checkpoint progress section.

        This message is emitted before persistence begins so a long-running
        checkpoint write is distinguishable from a stalled process.
        """
        logger.info("Saving checkpoint...\n")

    @staticmethod
    def log_checkpoint_complete(
        checkpoint_path: str,
        distributed: bool = False,
        best_path: str | None = None,
    ) -> None:
        """Close a checkpoint section with its persisted output paths.

        Parameters
        ----------
        checkpoint_path : str
            Path of the checkpoint produced at this training boundary.
        distributed : bool, default False
            Whether progress tables include a process-rank column.
        best_path : str, optional
            Stable best-checkpoint path updated by this save, if any.
        """
        separator = "=" * LogManager.stdout_table_width(distributed)
        completion = f"Checkpoint saved: {checkpoint_path}"
        if best_path is not None:
            completion += f"\nBest checkpoint updated: {best_path}"

        # A missing closing boundary therefore signals an incomplete save
        logger.info("%s\n%s\n", completion, separator)

    @staticmethod
    def log_stdout_summary(
        log_row: Mapping[str, Any],
        data: Mapping[str, Any],
        watch: Any,
        tstamp: str,
        iteration: int,
        epoch: float | None,
        model_train: bool,
        rank: int | None,
        distributed: bool,
        main_process: bool,
        mode: str | None = None,
        total_iterations: int | None = None,
    ) -> None:
        """Emit the human-readable iteration summary to stdout.

        Parameters
        ----------
        log_row : Mapping[str, Any]
            Flat scalar row produced by :meth:`collect`.
        data : Mapping[str, Any]
            Original data products used to fetch common display metrics.
        watch : object
            Stopwatch manager with a ``time`` method.
        tstamp : str
            Timestamp string associated with the iteration.
        iteration : int
            Iteration counter.
        epoch : float | None
            Progress in the training loop measured in epochs.
        model_train : bool
            Whether the current model, if any, is in training mode.
        rank : int | None
            Current process rank. ``None`` indicates CPU/single-process mode.
        distributed : bool
            Whether distributed synchronization is active.
        main_process : bool
            Whether this process should print shared headers and blank lines.
        mode : str, optional
            Explicit processing mode shown in the timing header. By default,
            this is inferred as training or inference from ``model_train``.
        total_iterations : int, optional
            Total number of iterations in a bounded pass, used to show
            validation progress.
        """
        # Resolve the table layout before rendering its shared header
        proc = mode or ("train" if model_train else "inference")
        device = "GPU" if rank is not None else "CPU"
        keys = [f"Time ({proc})", f"{device} memory", "Loss", "Accuracy"]
        widths = list(LogManager.STDOUT_WIDTHS)
        if distributed:
            keys = ["Rank"] + keys
            widths = [LogManager.STDOUT_RANK_WIDTH] + widths

        if main_process:
            epoch_value = -1.0 if epoch is None else epoch
            header = "  | " + "| ".join(
                [f"{keys[i]:<{widths[i]}}" for i in range(len(keys))]
            )
            separator = "  |" + "+".join(["-" * (w + 1) for w in widths])
            iter_label = f"Iter. {iteration}"
            if mode == "validation":
                iter_label = f"Val. {iteration + 1}"
                if total_iterations is not None:
                    iter_label += f"/{total_iterations}"
            msg = f"{iter_label} (epoch {epoch_value:.3f}) @ {tstamp}\n"
            msg += header + "|\n"
            msg += separator + "|"
            logger.info(msg)

        # Keep distributed process rows associated with the same header
        if distributed:
            runtime.distributed_barrier()

        t_iter = watch.time("iteration").wall
        t_net = 0.0
        if "model_time" in log_row:
            t_net = watch.time("model").wall
        net_fraction = 0.0 if t_iter == 0.0 else 100 * t_net / t_iter

        if rank is not None:
            mem, mem_perc = log_row["gpu_mem"], log_row["gpu_mem_perc"]
        else:
            mem, mem_perc = log_row["cpu_mem"], log_row["cpu_mem_perc"]

        acc = data.get("accuracy", -1.0)
        loss = data.get("loss", -1.0)
        values = [
            f"{t_iter:0.2f} s ({net_fraction:0.2f} %)",
            f"{mem:0.2f} GB ({mem_perc:0.2f} %)",
            f"{loss:0.3f}",
            f"{acc:0.3f}",
        ]
        if distributed:
            values = [f"{rank}"] + values

        # Render the local process row, then gather it for ordered rank output
        msg = "  | " + "| ".join(
            [f"{values[i]:<{widths[i]}}" for i in range(len(keys))]
        )
        msg += "|"
        if distributed:
            rows = runtime.distributed_all_gather_object((rank, msg))
            if main_process:
                for _, row_msg in sorted(
                    rows, key=lambda item: -1 if item[0] is None else item[0]
                ):
                    logger.info(row_msg)
                logger.info("")
            return

        logger.info(msg)
        if main_process:
            logger.info("")

    def close(self) -> None:
        """Flush and close all owned logging backends."""
        self.csv_logger.close()
        if self.tb_logger is not None:
            self.tb_logger.flush()
            self.tb_logger.close()


def get_first_entry(index: Any) -> Any:
    """Return the first entry identifier from a scalar or sequence index."""
    if isinstance(index, (list, tuple)):
        return index[0]
    if isinstance(index, np.ndarray) and index.ndim > 0:
        return index[0]
    return index
