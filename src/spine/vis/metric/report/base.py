"""Shared contracts and streaming helpers for metric report recipes.

Report recipes deliberately separate reduction from presentation. A reducer
reads one or more CSV shards and returns JSON-serializable sufficient
statistics; a renderer consumes only those statistics. This guarantees that
the numbers shown in a plot are identical to those stored in ``summary.json``
and allows plots to be regenerated without rerunning reconstruction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.vis.metric.distribution import histogram_quantiles

REPORT_SCHEMA_VERSION = "1.2.0"
DEFAULT_CHUNKSIZE = 100_000


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide two arrays while returning zero for empty denominator bins.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values.
    denominator : np.ndarray
        Denominator values broadcast-compatible with ``numerator``.

    Returns
    -------
    np.ndarray
        Floating-point ratios, with zero wherever the denominator is zero.
    """
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def distribution_summary(
    histogram: np.ndarray,
    edges: np.ndarray,
    *,
    count: int,
    value_sum: float,
    value_sum_sq: float,
) -> dict[str, Any]:
    """Build JSON-safe descriptive statistics from streaming accumulators.

    Parameters
    ----------
    histogram : np.ndarray
        Number of entries in each histogram bin.
    edges : np.ndarray
        Histogram edges, with one more element than ``histogram``.
    count : int
        Number of finite values accumulated.
    value_sum : float
        Sum of the accumulated values.
    value_sum_sq : float
        Sum of the squared accumulated values.

    Returns
    -------
    dict
        Count, mean, standard deviation, approximate quantiles and histogram
        counts. Empty distributions use ``None`` for undefined statistics.

    Notes
    -----
    Quantiles are estimated from the histogram because point-level values are
    intentionally not retained by streaming reducers.
    """
    mean = value_sum / count if count else None
    variance = max(value_sum_sq / count - mean**2, 0.0) if count else None
    return {
        "count": count,
        "mean": mean,
        "std": float(np.sqrt(variance)) if variance is not None else None,
        "quantiles": histogram_quantiles(histogram, edges),
        "histogram": histogram.tolist(),
    }


def event_columns(columns: Iterable[str]) -> list[str]:
    """Return the strongest event-identity columns available in a CSV.

    The run/subrun/event triplet is preferred when complete. Otherwise the
    loader-level ``file_index`` and ``index`` columns are used when present.

    Parameters
    ----------
    columns : iterable of str
        CSV column names.

    Returns
    -------
    list of str
        Ordered columns which jointly identify an event, or an empty list if
        the CSV does not carry event identity.
    """
    available = set(columns)
    run_columns = [name for name in ("run", "subrun", "event") if name in available]
    if len(run_columns) == 3:
        return run_columns
    return [name for name in ("file_index", "index") if name in available]


class InputCounts:
    """Track event and source-file identities without retaining metric rows.

    Event and data-file identities are namespaced by CSV path. This avoids
    accidentally merging equal local indices from independent analyzer
    shards while keeping memory use proportional to identities rather than
    point- or object-level rows.
    """

    def __init__(self) -> None:
        self.events: set[tuple[Any, ...]] = set()
        self.files: set[tuple[str, Any]] = set()

    def update(self, path: Path, chunk: pd.DataFrame) -> None:
        """Add identities found in one CSV chunk.

        Parameters
        ----------
        path : Path
            CSV shard containing the chunk.
        chunk : pd.DataFrame
            Rows returned by one chunked ``pandas.read_csv`` iteration.
        """
        columns = event_columns(chunk.columns)
        if columns:
            values = chunk[columns].drop_duplicates().itertuples(index=False, name=None)
            self.events.update((str(path), *value) for value in values)
        if "file_index" in chunk:
            self.files.update((str(path), value) for value in chunk.file_index.unique())

    def as_dict(self, paths: Sequence[Path], row_count: int) -> dict[str, int]:
        """Return serializable source counts for a recipe summary.

        Parameters
        ----------
        paths : sequence of Path
            All CSV shards represented by this counter.
        row_count : int
            Total number of rows read from those shards.

        Returns
        -------
        dict
            Counts of CSV shards, rows, distinct events and distinct input
            data files.
        """
        return {
            "csv_shards": len(paths),
            "rows": row_count,
            "events": len(self.events),
            "data_files": len(self.files),
        }


class ReportRecipe(ABC):
    """Two-operation interface shared by all report recipes.

    Subclasses implement :meth:`reduce` to produce a fully serializable
    summary and :meth:`render` to reconstruct figures from that summary. They
    must not calculate independent metric values during rendering.

    Attributes
    ----------
    key : str
        User-defined metric key used to name report artifacts.
    config : dict
        Recipe-specific configuration copied from the report YAML.
    """

    name: str

    def __init__(self, key: str, config: Mapping[str, Any]) -> None:
        """Initialize a report recipe.

        Parameters
        ----------
        key : str
            User-defined key of the metric configuration.
        config : mapping
            Recipe-specific report configuration.
        """
        self.key = key
        self.config = dict(config)

    @abstractmethod
    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce CSV shards to a fully serializable summary dictionary.

        Parameters
        ----------
        csv_paths : mapping of str to sequence of Path
            Discovered CSV shards grouped by configured input name.

        Returns
        -------
        dict
            JSON-serializable sufficient statistics and metric values.
        """

    @abstractmethod
    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render plots using only values present in ``summary``.

        Parameters
        ----------
        summary : mapping
            Result returned by :meth:`reduce`.
        output_dir : Path
            Directory in which to write figure files.
        formats : sequence of str
            Graphical output suffixes, such as ``png`` or ``pdf``.

        Returns
        -------
        list of Path
            Figure paths written by the renderer.
        """
