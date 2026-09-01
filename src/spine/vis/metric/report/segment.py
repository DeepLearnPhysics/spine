"""Streaming reduction and rendering of semantic confusion counts.

The semantic evaluation analyzer writes event-wise ``count_ij`` columns, where
``i`` is the predicted class and ``j`` the true class. Summing these columns is
a compact sufficient statistic, so this recipe never needs voxel-level output
or an in-memory concatenation of event records.
"""

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
    """Incrementally sum event-wise semantic confusion counts.

    The recipe derives class labels from :mod:`spine.constants`. ``classes``
    may restrict the matrix to selected semantic types, while
    ``class_mapping`` may pool multiple source types under a new display name.
    ``num_classes`` can enforce the expected raw matrix size and ``chunksize``
    controls bounded CSV reads.

    Both raw and aggregated matrices use predicted class on rows and true
    class on columns. ``excluded_count`` records entries discarded by a class
    restriction; mapped entries remain included.
    """

    name = "segment_confusion"
    _column = re.compile(r"^count_(\d)(\d)$")

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Sum ``count_ij`` columns without concatenating event rows.

        Parameters
        ----------
        csv_paths : mapping of str to sequence of Path
            Semantic summary CSV shards under the ``source`` input name.

        Returns
        -------
        dict
            Serializable confusion counts, class definitions, per-class
            precision/recall, global accuracy and input counts.

        Raises
        ------
        ValueError
            If no count columns exist or their inferred class count exceeds a
            configured ``num_classes``.
        """
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
                # Column discovery is repeated per shard chunk so malformed or
                # schema-inconsistent analyzer output fails at its origin.
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

        # Axis 0 is prediction and axis 1 is truth. Consequently support is a
        # column sum, while the predicted population is a row sum.
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
        """Render the confusion matrix represented in the summary.

        Parameters
        ----------
        summary : mapping
            Serialized result returned by :meth:`reduce`.
        output_dir : Path
            Destination directory for the confusion figure.
        formats : sequence of str
            Graphical file formats to write.

        Returns
        -------
        list of Path
            Paths of the generated confusion-matrix files.
        """
        figure = plot_confusion_matrix(summary["matrix"], summary["class_names"])
        return save_figure(figure, output_dir / f"{self.key}_confusion", formats)
