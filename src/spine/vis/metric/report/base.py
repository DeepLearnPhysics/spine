"""Shared contracts and streaming helpers for metric report recipes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.vis.metric.style import histogram_quantiles

REPORT_SCHEMA_VERSION = "1.1.0"
DEFAULT_CHUNKSIZE = 100_000


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide arrays while returning zero for empty denominator bins."""
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
    """Build JSON-safe descriptive statistics from streaming accumulators."""
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
    """Return the strongest event-identity columns available in a CSV."""
    available = set(columns)
    run_columns = [name for name in ("run", "subrun", "event") if name in available]
    if len(run_columns) == 3:
        return run_columns
    return [name for name in ("file_index", "index") if name in available]


class InputCounts:
    """Track event and source-file identities without retaining metric rows."""

    def __init__(self) -> None:
        self.events: set[tuple[Any, ...]] = set()
        self.files: set[tuple[str, Any]] = set()

    def update(self, path: Path, chunk: pd.DataFrame) -> None:
        """Add identities found in one CSV chunk."""
        columns = event_columns(chunk.columns)
        if columns:
            values = chunk[columns].drop_duplicates().itertuples(index=False, name=None)
            self.events.update((str(path), *value) for value in values)
        if "file_index" in chunk:
            self.files.update((str(path), value) for value in chunk.file_index.unique())

    def as_dict(self, paths: Sequence[Path], row_count: int) -> dict[str, int]:
        """Return serializable source counts for a recipe summary."""
        return {
            "csv_shards": len(paths),
            "rows": row_count,
            "events": len(self.events),
            "data_files": len(self.files),
        }


class ReportRecipe(ABC):
    """Two-operation interface shared by all report recipes."""

    name: str

    def __init__(self, key: str, config: Mapping[str, Any]) -> None:
        self.key = key
        self.config = dict(config)

    @abstractmethod
    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce CSV shards to a fully serializable summary dictionary."""

    @abstractmethod
    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render plots using only values present in ``summary``."""
