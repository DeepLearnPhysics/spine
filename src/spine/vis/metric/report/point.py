"""Point-proposal distance reduction and notebook-style rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.constants import LOWES_SHP
from spine.vis.metric.style import (
    plot_histogram_with_boxplot,
    plotting,
    save_figure,
)

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe, distribution_summary
from .classification import resolve_class_groups


class PointProposalRecipe(ReportRecipe):
    """Stream bidirectional PPN distances into curves and distributions."""

    name = "point_proposal"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce point rows in chunks, including optional per-shape summaries."""
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
            )

        return {
            "recipe": self.name,
            "inputs": {"csv_shards": len(all_paths)},
            "distance_thresholds": thresholds.tolist(),
            "distance_unit": self.config.get("distance_unit", "cm"),
            "histogram_edges": edges.tolist(),
            "classes": classes,
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
    ) -> dict[str, Any]:
        """Reduce one matching direction into threshold and histogram statistics."""
        total = matched = correct_type = 0
        value_sum = value_sum_sq = 0.0
        passing = np.zeros(len(thresholds), dtype=np.int64)
        histogram = np.zeros(len(edges) - 1, dtype=np.int64)
        class_histograms = {
            value["name"]: np.zeros(len(edges) - 1, dtype=np.int64) for value in classes
        }
        class_counts = {value["name"]: [0, 0.0, 0.0] for value in classes}
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
                valid = np.isfinite(distances) & (distances >= 0.0)
                selected = distances[valid]
                total += len(distances)
                matched += len(selected)
                value_sum += float(selected.sum())
                value_sum_sq += float(np.square(selected).sum())
                passing += [
                    np.count_nonzero(valid & (distances <= value))
                    for value in thresholds
                ]
                histogram += np.histogram(selected, bins=edges)[0]

                if {"shape", "closest_shape"} <= set(chunk.columns):
                    correct_type += int(
                        np.count_nonzero(
                            valid & (chunk["shape"] == chunk["closest_shape"])
                        )
                    )
                if classes and "shape" in chunk:
                    shapes = chunk["shape"].to_numpy(dtype=np.int64)
                    for group in classes:
                        name = group["name"]
                        values = distances[valid & np.isin(shapes, group["source_ids"])]
                        class_histograms[name] += np.histogram(values, bins=edges)[0]
                        class_counts[name][0] += len(values)
                        class_counts[name][1] += float(values.sum())
                        class_counts[name][2] += float(np.square(values).sum())
                counts.update(path, chunk)
                row_count += len(chunk)

        distribution = distribution_summary(
            histogram,
            edges,
            count=matched,
            value_sum=value_sum,
            value_sum_sq=value_sum_sq,
        )
        by_class = {
            name: distribution_summary(
                class_histograms[name],
                edges,
                count=int(class_counts[name][0]),
                value_sum=class_counts[name][1],
                value_sum_sq=class_counts[name][2],
            )
            for name in class_histograms
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
        """Render threshold curves and distance distributions from the summary."""
        paths = []
        plt = plotting()
        thresholds = np.asarray(summary["distance_thresholds"])
        unit = summary["distance_unit"]

        # Separate threshold curves remain legible in compact batch reports.
        for label in ("efficiency", "purity"):
            fractions = summary["directions"][label]["threshold_fraction"]
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
                by_class,
                summary["histogram_edges"],
                x_label=f"Closest-point distance [{unit}]",
            )
            paths.extend(
                save_figure(fig, output_dir / f"ppn_{label}_by_class", formats)
            )
        return paths
