"""Streaming reduction and rendering of point-proposal metrics.

Point-proposal analyzer records contain the closest-match distance in both
truth-to-reconstruction and reconstruction-to-truth directions. The two
directions respectively define efficiency and purity as functions of a
distance threshold, while their valid distances provide resolution summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.constants import LOWES_SHP
from spine.vis.metric.distribution import plot_histogram_with_boxplot
from spine.vis.metric.plot import plotting, save_figure

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe, distribution_summary
from .classification import resolve_class_groups


class PointProposalRecipe(ReportRecipe):
    """Stream bidirectional PPN distances into curves and distributions.

    Configuration supports ``distance_thresholds``, ``distance_range``,
    ``distance_scale``, ``distance_unit``, ``bins``, semantic ``classes`` or
    ``class_mapping``, ``overall_classes``, and ``chunksize``. The configured
    ``classes`` define the per-class curves, while ``overall_classes`` may
    independently restrict the aggregate population. If the latter is omitted,
    the aggregate includes every class represented by ``classes``. A negative
    or non-finite distance is treated as an unmatched point and contributes to
    the threshold denominator but not the resolution distribution.
    """

    name = "point_proposal"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce point rows in chunks, including per-shape summaries.

        Parameters
        ----------
        csv_paths : mapping of str to sequence of Path
            CSV shards under the required ``truth_to_reco`` and
            ``reco_to_truth`` input names.

        Returns
        -------
        dict
            Serializable thresholds, bin edges, classes and directional
            efficiency, purity and resolution statistics.
        """
        thresholds = np.asarray(
            self.config.get("distance_thresholds", (1.0, 2.0, 5.0)),
            dtype=np.float64,
        )
        if thresholds.ndim != 1 or not len(thresholds) or np.any(thresholds < 0):
            raise ValueError(
                "PPN distance thresholds must be a non-empty positive list."
            )
        thresholds.sort()

        distance_range = self.config.get("distance_range", [0.0, thresholds[-1]])
        bins = int(self.config.get("bins", 50))
        edges = np.linspace(
            float(distance_range[0]), float(distance_range[1]), bins + 1
        )
        scale = float(self.config.get("distance_scale", 1.0))
        classes = resolve_class_groups(
            self.config,
            kind="shape",
            default_ids=range(LOWES_SHP),
        )
        overall_classes = classes
        if "overall_classes" in self.config:
            overall_classes = resolve_class_groups(
                {"classes": self.config["overall_classes"]},
                kind="shape",
                default_ids=range(LOWES_SHP),
            )
        overall_ids = [
            source_id for group in overall_classes for source_id in group["source_ids"]
        ]
        directions = {}
        all_paths: list[Path] = []

        for source_key, label in (
            ("truth_to_reco", "efficiency"),
            ("reco_to_truth", "purity"),
        ):
            paths = list(csv_paths[source_key])
            all_paths.extend(paths)
            directions[label] = self._reduce_direction(
                paths,
                thresholds=thresholds,
                edges=edges,
                scale=scale,
                classes=classes,
                overall_ids=overall_ids,
            )

        return {
            "recipe": self.name,
            "inputs": {"csv_shards": len(all_paths)},
            "distance_thresholds": thresholds.tolist(),
            "distance_unit": self.config.get("distance_unit", "cm"),
            "histogram_edges": edges.tolist(),
            "classes": classes,
            "overall_classes": overall_classes,
            "directions": directions,
        }

    def _reduce_direction(
        self,
        paths: Sequence[Path],
        *,
        thresholds: np.ndarray,
        edges: np.ndarray,
        scale: float,
        classes: Sequence[Mapping[str, Any]],
        overall_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Reduce one matching direction into threshold and histogram statistics.

        Parameters
        ----------
        paths : sequence of Path
            CSV shards for one matching direction.
        thresholds : np.ndarray
            Sorted distance thresholds at which to count successful matches.
        edges : np.ndarray
            Fixed edges for overall and per-class distance histograms.
        scale : float
            Multiplicative conversion from stored distances to report units.
        classes : sequence of mappings
            Semantic groups used for per-class distance distributions.
        overall_ids : sequence of int
            Source shape IDs included in aggregate threshold statistics.

        Returns
        -------
        dict
            Input counts, threshold fractions, type accuracy and overall and
            per-class distance distributions.
        """
        total = matched = correct_type = 0
        value_sum = value_sum_sq = 0.0
        passing = np.zeros(len(thresholds), dtype=np.int64)
        histogram = np.zeros(len(edges) - 1, dtype=np.int64)
        class_histograms = {
            value["name"]: np.zeros(len(edges) - 1, dtype=np.int64) for value in classes
        }
        class_counts = {
            value["name"]: {
                "total": 0,
                "matched": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "passing": np.zeros(len(thresholds), dtype=np.int64),
            }
            for value in classes
        }
        counts = InputCounts()
        row_count = 0

        for path in paths:
            chunks = pd.read_csv(
                path,
                chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
            )
            for chunk in chunks:
                if "dist" not in chunk:
                    raise ValueError(f"Missing `dist` column in {path}.")
                distances = chunk.dist.to_numpy(dtype=np.float64) * scale
                if "shape" not in chunk:
                    raise ValueError(f"Missing `shape` column in {path}.")
                shapes = chunk["shape"].to_numpy(dtype=np.int64)

                # Class selection applies symmetrically to truth points for
                # efficiency and reconstructed points for purity.
                included = np.isin(shapes, overall_ids)
                # Analyzer sentinel distances are excluded from the resolution
                # distribution but remain in ``total`` for efficiency/purity.
                valid = included & np.isfinite(distances) & (distances >= 0.0)
                selected = distances[valid]
                total += int(np.count_nonzero(included))
                matched += len(selected)
                value_sum += float(selected.sum())
                value_sum_sq += float(np.square(selected).sum())
                passing += [
                    np.count_nonzero(valid & (distances <= value))
                    for value in thresholds
                ]
                histogram += np.histogram(selected, bins=edges)[0]

                if {"shape", "closest_shape"} <= set(chunk.columns):
                    # Type accuracy is conditional on a geometrically matched
                    # point and is therefore normalized by ``matched`` below.
                    correct_type += int(
                        np.count_nonzero(
                            valid & (chunk["shape"] == chunk["closest_shape"])
                        )
                    )
                for group in classes:
                    name = group["name"]
                    class_mask = np.isin(shapes, group["source_ids"])
                    class_valid = (
                        class_mask & np.isfinite(distances) & (distances >= 0.0)
                    )
                    values = distances[class_valid]
                    class_histograms[name] += np.histogram(values, bins=edges)[0]
                    class_counts[name]["total"] += int(np.count_nonzero(class_mask))
                    class_counts[name]["matched"] += len(values)
                    class_counts[name]["sum"] += float(values.sum())
                    class_counts[name]["sum_sq"] += float(np.square(values).sum())
                    class_counts[name]["passing"] += [
                        np.count_nonzero(class_valid & (distances <= threshold))
                        for threshold in thresholds
                    ]
                counts.update(path, chunk)
                row_count += len(chunk)

        distribution = distribution_summary(
            histogram,
            edges,
            count=matched,
            value_sum=value_sum,
            value_sum_sq=value_sum_sq,
        )
        by_class = {}
        for name, class_count in class_counts.items():
            class_total = class_count["total"]
            by_class[name] = {
                "total": class_total,
                "matched": class_count["matched"],
                "threshold_fraction": {
                    str(float(value)): (
                        float(count / class_total) if class_total else 0.0
                    )
                    for value, count in zip(thresholds, class_count["passing"])
                },
                "distribution": distribution_summary(
                    class_histograms[name],
                    edges,
                    count=class_count["matched"],
                    value_sum=class_count["sum"],
                    value_sum_sq=class_count["sum_sq"],
                ),
            }
        return {
            "inputs": counts.as_dict(paths, row_count),
            "total": total,
            "matched": matched,
            "threshold_fraction": {
                str(float(value)): float(count / total) if total else 0.0
                for value, count in zip(thresholds, passing)
            },
            "type_accuracy": float(correct_type / matched) if matched else 0.0,
            "distribution": distribution,
            "by_class": by_class,
        }

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render threshold curves and distance distributions from the summary.

        Parameters
        ----------
        summary : mapping
            Serialized result returned by :meth:`reduce`.
        output_dir : Path
            Destination directory for PPN figures.
        formats : sequence of str
            Graphical file formats to write.

        Returns
        -------
        list of Path
            Paths of efficiency, purity and resolution figures.
        """
        paths = []
        plt = plotting()
        thresholds = np.asarray(summary["distance_thresholds"])
        unit = summary["distance_unit"]

        # Separate threshold curves remain legible in compact batch reports.
        for label in ("efficiency", "purity"):
            direction = summary["directions"][label]
            fractions = direction["threshold_fraction"]
            values = [fractions[str(float(value))] for value in thresholds]
            fig, axis = plt.subplots(figsize=(8, 6))
            axis.plot(thresholds, values, marker="o", linewidth=2)
            axis.set(
                xlabel=f"Distance threshold [{unit}]",
                ylabel=label.capitalize(),
                ylim=(0.0, 1.02),
            )
            axis.grid(True)
            paths.extend(save_figure(fig, output_dir / f"ppn_{label}", formats))

            # A separate class-resolved curve keeps the aggregate plot clean
            # while making shape-dependent PPN behavior directly comparable.
            by_class = direction["by_class"]
            if by_class:
                fig, axis = plt.subplots(figsize=(8, 6))
                for class_name, class_summary in by_class.items():
                    fractions = class_summary["threshold_fraction"]
                    values = [fractions[str(float(value))] for value in thresholds]
                    axis.plot(
                        thresholds,
                        values,
                        marker="o",
                        linewidth=2,
                        label=class_name,
                    )
                axis.set(
                    xlabel=f"Distance threshold [{unit}]",
                    ylabel=label.capitalize(),
                    ylim=(0.0, 1.02),
                )
                axis.grid(True)
                axis.legend()
                paths.extend(
                    save_figure(
                        fig,
                        output_dir / f"ppn_{label}_by_class",
                        formats,
                    )
                )

        distributions = {
            "Closest prediction (efficiency)": summary["directions"]["efficiency"][
                "distribution"
            ],
            "Closest label (purity)": summary["directions"]["purity"]["distribution"],
        }
        fig = plot_histogram_with_boxplot(
            distributions,
            summary["histogram_edges"],
            x_label=f"Closest-point distance [{unit}]",
        )
        paths.extend(save_figure(fig, output_dir / "ppn_resolution", formats))

        for label in ("efficiency", "purity"):
            by_class = summary["directions"][label]["by_class"]
            if not by_class:
                continue
            fig = plot_histogram_with_boxplot(
                {
                    class_name: class_summary["distribution"]
                    for class_name, class_summary in by_class.items()
                },
                summary["histogram_edges"],
                x_label=f"Closest-point distance [{unit}]",
            )
            paths.extend(
                save_figure(
                    fig,
                    output_dir / f"ppn_resolution_{label}_by_class",
                    formats,
                )
            )
        return paths
