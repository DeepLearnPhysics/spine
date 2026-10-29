"""Optional event bootstrap for confusion-count reports.

Retain compact event matrices only when requested. Resample events with
replacement, using the same multiplicity in every distance bin. CSV paths
namespace local event IDs consistently with the report input counters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import event_columns
from .classification import classification_metrics


class EventBootstrap:
    """Collect event confusion counts and compute percentile intervals.

    Parameters
    ----------
    config : mapping
        Bootstrap options: ``replicates`` (default 1000), ``confidence``
        (default 0.95), and nonnegative integer ``seed`` (default 0).

    Notes
    -----
    Memory scales with events times bins times raw classes squared. Intervals
    describe event sampling variation, not training or systematic uncertainty.
    Undefined replicates are omitted and their valid count is reported. Fewer
    than two contributing events or valid replicates yields null endpoints.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("bootstrap must be a mapping of options.")
        unknown = set(config) - {"replicates", "confidence", "seed"}
        if unknown:
            raise ValueError(f"Unknown bootstrap options: {sorted(unknown)}")
        self.replicates = config.get("replicates", 1000)
        self.confidence = config.get("confidence", 0.95)
        self.seed = config.get("seed", 0)
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates < 2
        ):
            raise ValueError("bootstrap replicates must be an integer >= 2.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 < self.confidence < 1
        ):
            raise ValueError("bootstrap confidence must lie between 0 and 1.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("bootstrap seed must be a nonnegative integer.")
        self.events: dict[tuple, dict[int, np.ndarray]] = {}

    def update(
        self, path: Path, chunk: pd.DataFrame, size: int, *, distance: bool = False
    ) -> None:
        """Accumulate all rows for each event across chunk boundaries.

        Parameters
        ----------
        path : Path
            CSV shard used to namespace local event IDs.
        chunk : pandas.DataFrame
            Confusion-count rows with event identity columns.
        size : int
            Raw confusion matrix dimension.
        distance : bool, default False
            Use the row's ``bin_index``; otherwise use a single bin zero.

        Raises
        ------
        ValueError
            If event identities are absent or null, or counts are invalid.
        """
        identities = event_columns(chunk.columns)
        if not identities or chunk[identities].isna().any().any():
            raise ValueError(
                "Event bootstrap requires non-null event identity columns."
            )
        columns = [f"count_{i}{j}" for i in range(size) for j in range(size)]
        values = chunk.reindex(columns=columns, fill_value=0).to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values != np.floor(values))
        ):
            raise ValueError("Bootstrap confusion counts must be nonnegative integers.")
        bins = chunk.bin_index.to_numpy() if distance else np.zeros(len(chunk), int)
        keys = chunk[identities].itertuples(index=False, name=None)
        for identity, index, row in zip(keys, bins, values):
            event = self.events.setdefault((str(path), *identity), {})
            matrix = event.setdefault(
                int(index), np.zeros((size, size), dtype=np.int64)
            )
            matrix += row.astype(np.int64).reshape(size, size)

    def summarize(
        self, groups: Sequence[Mapping[str, Any]], bins: Sequence[int]
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        """Resample whole events and compute pointwise percentile intervals.

        Parameters
        ----------
        groups : sequence of mappings
            Report class definitions shared with the point estimates.
        bins : sequence of int
            Ordered bin IDs, including the no-vertex bin when applicable.

        Returns
        -------
        metadata : dict
            Method, sampling unit, confidence level, seed and event count.
        intervals : dict
            Bin-indexed intervals for accuracy, macro F1 and each class metric.
            Each interval records lower/upper endpoints and valid replicates.
        """
        events = list(self.events.values())
        count = len(events)
        metadata = {
            "method": "percentile",
            "unit": "event",
            "confidence": self.confidence,
            "replicates": self.replicates,
            "seed": self.seed,
            "events": count,
        }
        if not events:
            raise ValueError("Event bootstrap requires at least one event.")
        size = next(iter(events[0].values())).shape[0]
        data = np.zeros((count, len(bins), size, size), dtype=np.int64)
        for e, event in enumerate(events):
            for b, index in enumerate(bins):
                if index in event:
                    data[e, b] = event[index]
        samples = {index: [] for index in bins}
        rng = np.random.default_rng(self.seed)
        for _ in range(self.replicates):
            # One draw per replicate preserves event correlations between bins.
            weights = np.bincount(rng.integers(count, size=count), minlength=count)
            totals = np.einsum("e,ebij->bij", weights, data)
            for index, matrix in zip(bins, totals):
                samples[index].append(classification_metrics(matrix, groups))

        intervals = {}
        tail = (1 - self.confidence) / 2
        for b, index in enumerate(bins):
            point = classification_metrics(data[:, b].sum(axis=0), groups)
            # A single populated event cannot estimate between-event variation.
            contributing = int(np.count_nonzero(data[:, b].sum(axis=(1, 2))))

            event_metrics = [
                classification_metrics(matrix, groups) for matrix in data[:, b]
            ]

            def interval(values, estimate, contributing_events):
                """Summarize defined replicate values for one scalar metric."""
                valid = [value for value in values if value is not None]
                endpoints = [None, None]
                if (
                    estimate is not None
                    and contributing_events >= 2
                    and len(valid) >= 2
                ):
                    endpoints = np.quantile(valid, [tail, 1 - tail]).tolist()
                return {
                    "lower": endpoints[0],
                    "upper": endpoints[1],
                    "valid_replicates": len(valid),
                    "contributing_events": contributing_events,
                }

            results = samples[index]
            intervals[index] = {
                "contributing_events": contributing,
                **{
                    metric: interval(
                        [s[metric] for s in results],
                        point[metric],
                        sum(s[metric] is not None for s in event_metrics),
                    )
                    for metric in ("accuracy", "macro_f1")
                },
                "per_class": {
                    group["name"]: {
                        metric: interval(
                            [s["per_class"][group["name"]][metric] for s in results],
                            point["per_class"][group["name"]][metric],
                            sum(
                                s["per_class"][group["name"]][metric] is not None
                                for s in event_metrics
                            ),
                        )
                        for metric in ("precision", "recall", "f1")
                    }
                    for group in groups
                },
            }
        return metadata, intervals
