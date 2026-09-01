"""Semantic-segmentation confusion reduction and rendering."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.vis.metric.style import plot_confusion_matrix, save_figure

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe, safe_ratio
from .classification import aggregate_confusion, resolve_class_groups


class SegmentConfusionRecipe(ReportRecipe):
    """Incrementally sum event-wise semantic confusion counts."""

    name = "segment_confusion"
    _column = re.compile(r"^count_(\d)(\d)$")

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Sum ``count_ij`` columns without concatenating event rows."""
        paths = list(csv_paths["source"])
        num_classes = self.config.get("num_classes")
        matrix: np.ndarray | None = None
        counts = InputCounts()
        row_count = 0

        for path in paths:
            chunks = pd.read_csv(
                path,
                chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
            )
            for chunk in chunks:
                columns = {
                    match.groups(): column
                    for column in chunk.columns
                    if (match := self._column.match(column))
                }
                if not columns:
                    raise ValueError(f"No confusion count columns found in {path}.")

                inferred = max(max(int(i), int(j)) for i, j in columns) + 1
                size = int(num_classes or inferred)
                if inferred > size:
                    raise ValueError(
                        f"Configured {size} classes, but {path} contains {inferred}."
                    )
                if matrix is None:
                    matrix = np.zeros((size, size), dtype=np.int64)
                for (prediction, truth), column in columns.items():
                    matrix[int(prediction), int(truth)] += int(chunk[column].sum())
                counts.update(path, chunk)
                row_count += len(chunk)

        assert matrix is not None
        raw_total = int(matrix.sum())
        classes = resolve_class_groups(
            self.config,
            kind="shape",
            default_ids=range(len(matrix)),
        )
        matrix = aggregate_confusion(matrix, classes)
        class_names = [value["name"] for value in classes]

        support = matrix.sum(axis=0)
        predicted = matrix.sum(axis=1)
        diagonal = np.diag(matrix)
        recall = safe_ratio(diagonal, support)
        precision = safe_ratio(diagonal, predicted)
        return {
            "recipe": self.name,
            "inputs": counts.as_dict(paths, row_count),
            "classes": classes,
            "class_names": class_names,
            "matrix": matrix.tolist(),
            "excluded_count": raw_total - int(matrix.sum()),
            "per_class": {
                name: {
                    "support": int(support[index]),
                    "predicted": int(predicted[index]),
                    "recall": float(recall[index]),
                    "precision": float(precision[index]),
                }
                for index, name in enumerate(class_names)
            },
            "accuracy": float(diagonal.sum() / matrix.sum()) if matrix.sum() else 0.0,
        }

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render the confusion matrix represented in the summary."""
        figure = plot_confusion_matrix(summary["matrix"], summary["class_names"])
        return save_figure(figure, output_dir / f"{self.key}_confusion", formats)
