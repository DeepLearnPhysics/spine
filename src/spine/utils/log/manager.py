"""Structured scalar logging manager."""

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import psutil

from spine.io.write.csv import CSVWriter
from spine.utils.logger import logger
from spine.utils.torch import runtime

__all__ = ["LogManager"]


class LogManager:
    """Manage structured scalar logs for a driver-like processing loop.

    The manager writes one flat scalar row to CSV on every call and can mirror
    numeric entries to TensorBoard. It also owns the human-readable progress
    table printed periodically during training or inference.
    """

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
        self.csv_logger = CSVWriter(
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
        """
        proc = "train" if model_train else "inference"
        device = "GPU" if rank is not None else "CPU"
        keys = [f"Time ({proc})", f"{device} memory", "Loss", "Accuracy"]
        widths = [20, 20, 9, 9]
        if distributed:
            keys = ["Rank"] + keys
            widths = [5] + widths
        if main_process:
            epoch_value = -1.0 if epoch is None else epoch
            header = "  | " + "| ".join(
                [f"{keys[i]:<{widths[i]}}" for i in range(len(keys))]
            )
            separator = "  |" + "+".join(["-" * (w + 1) for w in widths])
            msg = f"Iter. {iteration} (epoch {epoch_value:.3f}) @ {tstamp}\n"
            msg += header + "|\n"
            msg += separator + "|"
            logger.info(msg)
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

        msg = "  | " + "| ".join(
            [f"{values[i]:<{widths[i]}}" for i in range(len(keys))]
        )
        msg += "|"
        logger.info(msg)

        if distributed:
            runtime.distributed_barrier()
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
