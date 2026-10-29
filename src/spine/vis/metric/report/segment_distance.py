"""Streaming distance-dependent segmentation accuracy and class recall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.vis.metric.plot import plotting, save_figure

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe
from .classification import aggregate_confusion, resolve_class_groups
from .segment import SegmentConfusionRecipe


class SegmentDistanceRecipe(ReportRecipe):
    """Reduce distance confusion counts; empty denominators serialize as null.

    Supports the same ``classes`` and ``class_mapping`` as segment_confusion.
    Truth denominators include predictions outside selected classes. Distance
    intervals are in cm; a maximum of null denotes the unbounded final bin.
    """

    name = "segment_distance"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Sum event counts before computing distance-dependent metrics.

        Parameters
        ----------
        csv_paths : mapping of str to sequence of Path
            Analyzer CSV shards under the ``source`` input name. Each shard
            must use the same vertex product, class layout and bin edges.

        Returns
        -------
        dict
            JSON-compatible bin bounds, raw confusion matrices, overall
            accuracy and per-class recall. A separate ``no_vertex`` summary
            includes the excluded event count and its classification metrics.
            Empty denominators and unbounded upper edges serialize as null.

        Raises
        ------
        ValueError
            If count schemas or vertex products differ, counts are invalid,
            or distance bins are missing, inconsistent or noncontiguous.

        Notes
        -----
        Metrics are voxel-weighted, rather than averages of event accuracies.
        Class selections restrict the truth population but retain predictions
        outside that selection in the denominator.
        """
        paths = list(csv_paths["source"])
        counts = InputCounts()
        matrices, bounds = {}, {}
        schema = None
        vertex_key = None
        row_count = no_vertex_events = 0

        # Stream shards while checking that their statistics can be pooled.
        for path in paths:
            for chunk in pd.read_csv(
                path, chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE)
            ):
                # Require a complete, consistent predicted-by-true matrix.
                columns = sorted(
                    c for c in chunk if SegmentConfusionRecipe._column.match(c)
                )
                size = max((int(c[-1]) for c in columns), default=-1) + 1
                expected = [f"count_{i}{j}" for i in range(size) for j in range(size)]
                if (
                    not columns
                    or columns != expected
                    or (schema is not None and columns != schema)
                ):
                    raise ValueError(
                        f"Inconsistent distance confusion schema in {path}."
                    )
                if self.config.get("num_classes", size) != size:
                    raise ValueError(
                        "Configured num_classes differs from distance counts."
                    )
                schema = columns

                # A report describes distances to one configured point set.
                keys = chunk.vertex_key.unique()
                if len(keys) != 1 or (vertex_key is not None and vertex_key != keys[0]):
                    raise ValueError("Cannot combine different vertex products.")
                vertex_key = keys[0]

                values = chunk[columns].to_numpy(dtype=float)
                if (
                    not np.all(np.isfinite(values))
                    or np.any(values < 0)
                    or np.any(values != np.floor(values))
                ):
                    raise ValueError(
                        "Distance confusion counts must be nonnegative integers."
                    )

                # Count events independently of the repeated distance-bin rows.
                counts.update(path, chunk)
                row_count += len(chunk)
                no_vertex_events += int(
                    ((chunk.bin_index == -1) & (chunk.num_vertices == 0)).sum()
                )

                # Verify interval definitions before adding their counts.
                for index, group in chunk.groupby("bin_index"):
                    if index != int(index) or index < -1:
                        raise ValueError("Invalid distance bin index.")
                    index = int(index)
                    intervals = group[
                        ["distance_min", "distance_max"]
                    ].drop_duplicates()
                    interval = tuple(intervals.iloc[0])
                    if len(intervals) != 1 or (
                        index in bounds and bounds[index] != interval
                    ):
                        raise ValueError(
                            "Inconsistent distance bin edges across shards."
                        )

                    bounds[index] = interval
                    matrices.setdefault(index, np.zeros((size, size), dtype=np.int64))
                    matrices[index] += (
                        group[columns]
                        .sum()
                        .to_numpy(dtype=np.int64)
                        .reshape(size, size)
                    )

        # Check the complete interval sequence after reading every shard.
        if schema is None or sorted(matrices) != list(range(-1, len(matrices) - 1)):
            raise ValueError("Missing distance bins.")
        if bounds[-1] != (-1, -1) or bounds[0][0] != 0:
            raise ValueError("Invalid no-vertex or first distance bin.")

        for index in range(len(matrices) - 1):
            lower, upper = bounds[index]
            last = index == len(matrices) - 2
            if (
                not np.isfinite(lower)
                or (last and upper != -1)
                or (not last and (upper <= lower or upper != bounds[index + 1][0]))
            ):
                raise ValueError(
                    "Distance bins must be contiguous and end in overflow."
                )

        # Apply display selections only after aggregating the raw matrices.
        classes = resolve_class_groups(
            self.config, kind="shape", default_ids=range(size)
        )

        def summarize(matrix):
            """Derive voxel-weighted metrics from one raw confusion matrix.

            Rows represent predicted classes and columns represent truth.
            Class mapping pools correct counts, while support includes every
            prediction for the selected true classes. Empty populations yield
            None so JSON and plots preserve undefined accuracies.
            """
            support = [int(matrix[:, c["source_ids"]].sum()) for c in classes]
            pooled = aggregate_confusion(matrix, classes)
            correct = np.diag(pooled)
            total = sum(support)

            return {
                "matrix": matrix.tolist(),
                "voxel_count": total,
                "accuracy": float(correct.sum() / total) if total else None,
                "per_class": {
                    c["name"]: {"support": n, "recall": float(k / n) if n else None}
                    for c, n, k in zip(classes, support, correct)
                },
            }

        return {
            "recipe": self.name,
            "inputs": counts.as_dict(paths, row_count),
            "vertex_key": vertex_key,
            "distance_units": "cm",
            "classes": classes,
            "bins": [
                {
                    "distance_min": bounds[i][0],
                    "distance_max": bounds[i][1] if bounds[i][1] >= 0 else None,
                    **summarize(matrices[i]),
                }
                for i in sorted(matrices)
                if i >= 0
            ],
            "no_vertex": {"event_count": no_vertex_events, **summarize(matrices[-1])},
        }

    def render(
        self, summary: Mapping[str, Any], output_dir: Path, formats: Sequence[str]
    ) -> list[Path]:
        """Plot accuracy and recall above their distance-bin populations.

        Parameters
        ----------
        summary : mapping
            Serialized output of :meth:`reduce`, including class definitions,
            interval bounds and metrics for each distance bin.
        output_dir : Path
            Destination directory for the generated figures.
        formats : sequence of str
            Graphical formats accepted by :func:`save_figure`.

        Returns
        -------
        list of Path
            Paths of the saved distance-report figures.

        Notes
        -----
        Intervals use equally spaced positions and explicit cm labels, which
        also accommodates the unbounded final bin. Undefined metrics produce
        gaps; no-vertex statistics remain in the JSON summary rather than
        appearing at an arbitrary distance.
        """
        plt = plotting()
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
        bins = summary["bins"]
        positions = np.arange(len(bins))

        # Show the overall metric and one recall series per selected class.
        axes[0].plot(
            positions, [b["accuracy"] for b in bins], "o-", label="Overall accuracy"
        )
        for cls in summary["classes"]:
            name = cls["name"]
            axes[0].plot(
                positions,
                [b["per_class"][name]["recall"] for b in bins],
                "o-",
                label=f"{name} recall",
            )

        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Accuracy / recall")
        axes[0].legend()

        # Keep the voxel population visible when interpreting sparse bins.
        axes[1].bar(positions, [b["voxel_count"] for b in bins])
        labels = [
            (
                f"[{b['distance_min']:g}, {b['distance_max']:g})"
                if b["distance_max"] is not None
                else f"≥ {b['distance_min']:g}"
            )
            for b in bins
        ]

        axes[1].set_xticks(positions, labels, rotation=30)
        axes[1].set_ylabel("Voxels")
        axes[1].set_xlabel(f"Distance to closest {summary['vertex_key']} point (cm)")

        figure.tight_layout()
        return save_figure(figure, output_dir / f"{self.key}_distance", formats)
