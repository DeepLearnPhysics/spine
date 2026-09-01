"""Streaming reduction and rendering of clustering quality metrics.

The clustering analyzer writes one row per event at the fragment, particle
and interaction levels. This module accumulates bounded histograms and scalar
moments for each configured metric, optionally broken down by semantic class,
without concatenating all analyzer output in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.constants import LOWES_SHP
from spine.vis.metric.distribution import plot_histogram_with_boxplot
from spine.vis.metric.plot import save_figure

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe, distribution_summary
from .classification import resolve_class_groups


class ClusterSummaryRecipe(ReportRecipe):
    """Stream fragment, particle and interaction clustering distributions.

    Configuration accepts ``metric_names`` (default ``ari``, ``eff``,
    ``pur``), ``bins``, per-metric ``metric_ranges``, semantic ``classes`` or
    ``class_mapping``, and ``chunksize``. Named input sources are treated as
    clustering levels; interaction inputs omit the semantic breakdown because
    their analyzer records do not contain per-shape columns.

    The serialized result stores overall and per-class distributions under
    ``levels``. Each distribution contains sufficient statistics and bin
    edges needed to reproduce its plot exactly.
    """

    name = "cluster_summary"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce overall and per-shape clustering columns in bounded memory.

        Parameters
        ----------
        csv_paths : mapping of str to sequence of Path
            CSV shards grouped by clustering level, typically ``fragment``,
            ``particle`` and ``interaction``.

        Returns
        -------
        dict
            Serializable class definitions and distributions for every level.
        """
        metric_names = list(self.config.get("metric_names", ("ari", "eff", "pur")))
        classes = resolve_class_groups(
            self.config,
            kind="shape",
            default_ids=range(LOWES_SHP),
        )
        bins = int(self.config.get("bins", 50))
        metric_ranges = {
            "ari": (-1.0, 1.0),
            "eff": (0.0, 1.0),
            "pur": (0.0, 1.0),
            **self.config.get("metric_ranges", {}),
        }
        levels = {}
        all_paths: list[Path] = []
        for level, path_values in csv_paths.items():
            paths = list(path_values)
            all_paths.extend(paths)
            levels[level] = self._reduce_level(
                paths,
                metric_names=metric_names,
                classes=classes if level != "interaction" else [],
                bins=bins,
                metric_ranges=metric_ranges,
            )
        return {
            "recipe": self.name,
            "inputs": {"csv_shards": len(all_paths)},
            "classes": classes,
            "levels": levels,
        }

    def _reduce_level(
        self,
        paths: Sequence[Path],
        *,
        metric_names: Sequence[str],
        classes: Sequence[Mapping[str, Any]],
        bins: int,
        metric_ranges: Mapping[str, Sequence[float]],
    ) -> dict[str, Any]:
        """Reduce one clustering aggregation level.

        Parameters
        ----------
        paths : sequence of Path
            CSV shards belonging to this aggregation level.
        metric_names : sequence of str
            Numeric analyzer columns to summarize.
        classes : sequence of mappings
            Semantic groups to pool in per-class distributions. An empty
            sequence disables the breakdown.
        bins : int
            Number of histogram bins per metric.
        metric_ranges : mapping
            Lower and upper histogram limits keyed by metric name.

        Returns
        -------
        dict
            Input counts plus overall and per-class metric distributions.
        """
        accumulators = {}
        class_accumulators: dict[str, dict[str, dict[str, Any]]] = {}
        # Each metric owns its edges because ARI and overlap metrics commonly
        # cover different numeric ranges.
        for metric in metric_names:
            metric_range = metric_ranges.get(metric, (0.0, 1.0))
            if len(metric_range) != 2:
                raise ValueError(
                    f"Clustering range for `{metric}` must contain two bounds."
                )
            lower, upper = metric_range
            edges = np.linspace(float(lower), float(upper), bins + 1)
            accumulators[metric] = self._new_accumulator(edges)
            class_accumulators[metric] = {
                group["name"]: self._new_accumulator(edges) for group in classes
            }

        counts = InputCounts()
        row_count = 0
        for path in paths:
            chunks = pd.read_csv(
                path,
                chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
            )
            for chunk in chunks:
                missing = set(metric_names) - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"Missing clustering columns {sorted(missing)} in {path}."
                    )
                for metric, accumulator in accumulators.items():
                    self._update_accumulator(accumulator, chunk[metric].to_numpy())
                    # Per-shape analyzer columns follow ``<metric>_<shape>``.
                    # Mapping several shapes into one group pools their
                    # sufficient statistics in the same accumulator.
                    for group in classes:
                        group_accumulator = class_accumulators[metric][group["name"]]
                        for source_id in group["source_ids"]:
                            column = f"{metric}_{source_id}"
                            if column in chunk:
                                self._update_accumulator(
                                    group_accumulator,
                                    chunk[column].to_numpy(),
                                )
                counts.update(path, chunk)
                row_count += len(chunk)

        overall = {
            metric: self._finish_accumulator(accumulator)
            for metric, accumulator in accumulators.items()
        }
        by_class = {
            group["name"]: {
                metric: self._finish_accumulator(
                    class_accumulators[metric][group["name"]]
                )
                for metric in metric_names
            }
            for group in classes
        }
        return {
            "inputs": counts.as_dict(paths, row_count),
            "metrics": overall,
            "by_class": by_class,
        }

    @staticmethod
    def _new_accumulator(
        edges: np.ndarray,
    ) -> dict[str, Any]:
        """Create mutable sufficient statistics for one numeric column.

        Parameters
        ----------
        edges : np.ndarray
            Fixed histogram edges for the distribution.

        Returns
        -------
        dict
            Mutable count, moments, histogram and edge storage.
        """
        return {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "histogram": np.zeros(len(edges) - 1, dtype=np.int64),
            "edges": edges,
        }

    @staticmethod
    def _update_accumulator(
        accumulator: dict[str, Any], raw_values: np.ndarray
    ) -> None:
        """Add finite values within the configured metric range.

        Values outside the histogram range are excluded from both the
        histogram and scalar moments so every serialized statistic describes
        the same population.

        Parameters
        ----------
        accumulator : dict
            Mutable state returned by :meth:`_new_accumulator`.
        raw_values : np.ndarray
            Values read from one CSV chunk.
        """
        values = np.asarray(raw_values, dtype=np.float64)
        lower, upper = accumulator["edges"][[0, -1]]
        values = values[np.isfinite(values) & (values >= lower) & (values <= upper)]
        accumulator["count"] += len(values)
        accumulator["sum"] += float(values.sum())
        accumulator["sum_sq"] += float(np.square(values).sum())
        accumulator["histogram"] += np.histogram(
            values,
            bins=accumulator["edges"],
        )[0]

    @staticmethod
    def _finish_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one mutable accumulator to a JSON-safe distribution.

        Parameters
        ----------
        accumulator : mapping
            Completed mutable sufficient statistics.

        Returns
        -------
        dict
            Distribution summary including JSON-safe histogram edges.
        """
        result = distribution_summary(
            accumulator["histogram"],
            accumulator["edges"],
            count=accumulator["count"],
            value_sum=accumulator["sum"],
            value_sum_sq=accumulator["sum_sq"],
        )
        result["histogram_edges"] = accumulator["edges"].tolist()
        return result

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render overall and per-class plots for every clustering level.

        Parameters
        ----------
        summary : mapping
            Serialized result returned by :meth:`reduce`.
        output_dir : Path
            Destination directory for clustering figures.
        formats : sequence of str
            Graphical file formats to write.

        Returns
        -------
        list of Path
            Paths of all generated figures.
        """
        paths = []
        for level, level_summary in summary["levels"].items():
            # Metrics may use different ranges (ARI commonly spans -1 to 1,
            # while efficiency and purity span 0 to 1). Render them separately
            # so each plot is reconstructed with its own recorded bin edges.
            for metric, values in level_summary["metrics"].items():
                paths.extend(
                    self._render_distribution(
                        values,
                        label=metric.upper(),
                        x_label=f"{level.capitalize()} clustering {metric}",
                        output_path=output_dir / f"clustering_{level}_{metric}",
                        formats=formats,
                    )
                )

            # Compare semantic classes on one axis, as in the notebook. All
            # classes for a given metric share its configured range and bins.
            by_class = level_summary["by_class"]
            for metric in level_summary["metrics"]:
                distributions = {
                    class_name: class_metrics[metric]
                    for class_name, class_metrics in by_class.items()
                    if metric in class_metrics
                }
                if not distributions:
                    continue
                values = next(iter(distributions.values()))
                figure = plot_histogram_with_boxplot(
                    distributions,
                    values["histogram_edges"],
                    x_label=f"{level.capitalize()} clustering {metric}",
                )
                paths.extend(
                    save_figure(
                        figure,
                        output_dir / f"clustering_{level}_{metric}_by_class",
                        formats,
                    )
                )
        return paths

    @staticmethod
    def _render_distribution(
        values: Mapping[str, Any],
        *,
        label: str,
        x_label: str,
        output_path: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render one distribution from exactly its serialized statistics.

        Parameters
        ----------
        values : mapping
            Distribution statistics, including ``histogram_edges``.
        label : str
            Legend label for the distribution.
        x_label : str
            Horizontal-axis label.
        output_path : Path
            Figure path without an extension.
        formats : sequence of str
            Graphical file formats to write.

        Returns
        -------
        list of Path
            Figure paths written by :func:`save_figure`.
        """
        figure = plot_histogram_with_boxplot(
            {label: values},
            values["histogram_edges"],
            x_label=x_label,
        )
        return save_figure(figure, output_path, formats)
