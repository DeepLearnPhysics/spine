"""Reduction of configurable node predictions stored by ``SaveAna``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spine.constants import ParticlePID, ParticleShape
from spine.vis.metric.style import (
    plot_confusion_matrix,
    plot_histogram_with_boxplot,
    save_figure,
)

from .base import DEFAULT_CHUNKSIZE, InputCounts, ReportRecipe, distribution_summary
from .classification import infer_class_kind, map_class_values, resolve_class_groups


def quality_cut_mask(
    frame: pd.DataFrame, specification: Mapping[str, Any] | None
) -> np.ndarray:
    """Evaluate a nested, declarative quality-cut specification.

    A specification may contain ``all`` or ``any`` lists and a unary ``not``
    expression for boolean composition. Leaf predicates name one ``column``
    and one or more of ``min``, ``max``, ``equals``, ``not_equals``, ``in``,
    ``not_in``, ``abs_equals`` or ``abs_not_equals``. Bounds are inclusive.
    """
    if not specification:
        return np.ones(len(frame), dtype=bool)
    if "not" in specification:
        return ~quality_cut_mask(frame, specification["not"])
    if "all" in specification:
        mask = np.ones(len(frame), dtype=bool)
        for child in specification["all"]:
            mask &= quality_cut_mask(frame, child)
        return mask
    if "any" in specification:
        mask = np.zeros(len(frame), dtype=bool)
        for child in specification["any"]:
            mask |= quality_cut_mask(frame, child)
        return mask

    column = specification.get("column")
    if column not in frame:
        raise ValueError(f"Quality-cut column `{column}` is missing from the save CSV.")
    values = frame[column].to_numpy()
    mask = np.ones(len(frame), dtype=bool)
    if "min" in specification:
        mask &= values >= specification["min"]
    if "max" in specification:
        mask &= values <= specification["max"]
    if "equals" in specification:
        mask &= values == specification["equals"]
    if "not_equals" in specification:
        mask &= values != specification["not_equals"]
    if "in" in specification:
        mask &= np.isin(values, specification["in"])
    if "not_in" in specification:
        mask &= ~np.isin(values, specification["not_in"])
    if "abs_equals" in specification:
        mask &= np.abs(values) == specification["abs_equals"]
    if "abs_not_equals" in specification:
        mask &= np.abs(values) != specification["abs_not_equals"]
    return mask


class NodeSummaryRecipe(ReportRecipe):
    """Evaluate classification and orientation tasks from matched save rows."""

    name = "node_summary"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        """Reduce every configured task while reading each source shard once."""
        tasks = self.config.get("tasks")
        if not isinstance(tasks, Mapping) or not tasks:
            raise ValueError("A node summary requires a non-empty `tasks` mapping.")
        unknown_sources = {task.get("source") for task in tasks.values()} - set(
            csv_paths
        )
        if unknown_sources:
            raise ValueError(
                "Node tasks reference undefined sources: " f"{sorted(unknown_sources)}."
            )

        states = {key: self._initialize_task(key, task) for key, task in tasks.items()}
        source_inputs = {}
        for source, paths_value in csv_paths.items():
            paths = list(paths_value)
            source_tasks = {
                key: task for key, task in tasks.items() if task.get("source") == source
            }
            counts = InputCounts()
            row_count = 0
            for path in paths:
                chunks = pd.read_csv(
                    path,
                    chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
                )
                for chunk in chunks:
                    for key, task in source_tasks.items():
                        mask = quality_cut_mask(chunk, task.get("quality_cuts"))
                        self._update_task(states[key], task, chunk.loc[mask])
                    counts.update(path, chunk)
                    row_count += len(chunk)
            source_inputs[source] = counts.as_dict(paths, row_count)

        return {
            "recipe": self.name,
            "inputs": source_inputs,
            "tasks": {key: self._finish_task(state) for key, state in states.items()},
        }

    @staticmethod
    def _initialize_task(key: str, task: Mapping[str, Any]) -> dict[str, Any]:
        """Create mutable state for one configured node task."""
        task_type = task.get("type", "classification")
        if task_type == "classification":
            truth_column = task.get("truth_column")
            if not truth_column:
                raise ValueError(f"Classification task `{key}` needs `truth_column`.")
            class_type = task.get("class_type") or infer_class_kind(truth_column)
            if class_type == "shape":
                default_ids = [int(value) for value in ParticleShape if int(value) >= 0]
            elif class_type == "pid":
                default_ids = [int(value) for value in ParticlePID if int(value) >= 0]
            elif class_type == "primary":
                default_ids = [0, 1]
            else:
                raise ValueError(f"Unknown class type `{class_type}` for `{key}`.")
            classes = resolve_class_groups(
                task,
                kind=class_type,
                default_ids=default_ids,
            )
            return {
                "type": task_type,
                "classes": classes,
                "class_names": [value["name"] for value in classes],
                "matrix": np.zeros((len(classes), len(classes)), dtype=np.int64),
                "selected_rows": 0,
            }
        if task_type == "orientation":
            bins = int(task.get("bins", 50))
            edges = np.linspace(*task.get("range", (-1.0, 1.0)), bins + 1)
            return {
                "type": task_type,
                "edges": edges,
                "histogram": np.zeros(bins, dtype=np.int64),
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "selected_rows": 0,
            }
        raise ValueError(f"Unknown node task type `{task_type}` for `{key}`.")

    def _update_task(
        self,
        state: dict[str, Any],
        task: Mapping[str, Any],
        frame: pd.DataFrame,
    ) -> None:
        """Add one selected CSV chunk to a task accumulator."""
        state["selected_rows"] += len(frame)
        if state["type"] == "classification":
            truth_column = task.get("truth_column")
            prediction_column = task.get("prediction_column")
            missing = {truth_column, prediction_column} - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Node classification columns are missing: {sorted(missing)}."
                )
            truth = frame[truth_column].to_numpy(dtype=np.int64)
            prediction = frame[prediction_column].to_numpy(dtype=np.int64)
            truth, truth_valid = map_class_values(truth, state["classes"])
            prediction, prediction_valid = map_class_values(
                prediction, state["classes"]
            )
            valid = truth_valid & prediction_valid
            size = len(state["class_names"])
            valid &= (truth < size) & (prediction < size)
            state["matrix"] += np.histogram2d(
                prediction[valid],
                truth[valid],
                bins=(size, size),
                range=((0, size), (0, size)),
            )[0].astype(np.int64)
            return

        truth_columns = list(task.get("truth_columns", []))
        prediction_columns = list(task.get("prediction_columns", []))
        missing = set(truth_columns + prediction_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Node orientation columns are missing: {sorted(missing)}."
            )
        truth = frame[truth_columns].to_numpy(dtype=np.float64)
        prediction = frame[prediction_columns].to_numpy(dtype=np.float64)
        denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
        valid = np.isfinite(truth).all(axis=1) & np.isfinite(prediction).all(axis=1)
        valid &= denominator > 0.0
        cosine = np.sum(truth[valid] * prediction[valid], axis=1) / denominator[valid]
        cosine = np.clip(cosine, -1.0, 1.0)
        state["histogram"] += np.histogram(cosine, bins=state["edges"])[0]
        state["count"] += len(cosine)
        state["sum"] += float(cosine.sum())
        state["sum_sq"] += float(np.square(cosine).sum())

    @staticmethod
    def _finish_task(state: Mapping[str, Any]) -> dict[str, Any]:
        """Convert mutable task state to a JSON-safe result."""
        if state["type"] == "classification":
            matrix = state["matrix"]
            return {
                "type": state["type"],
                "selected_rows": state["selected_rows"],
                "evaluated_rows": int(matrix.sum()),
                "classes": state["classes"],
                "class_names": state["class_names"],
                "matrix": matrix.tolist(),
                "accuracy": (
                    float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
                ),
            }
        distribution = distribution_summary(
            state["histogram"],
            state["edges"],
            count=state["count"],
            value_sum=state["sum"],
            value_sum_sq=state["sum_sq"],
        )
        distribution["histogram_edges"] = state["edges"].tolist()
        return {
            "type": state["type"],
            "selected_rows": state["selected_rows"],
            "evaluated_rows": distribution["count"],
            "forward_fraction": NodeSummaryRecipe._forward_fraction(
                state["histogram"],
                state["edges"],
            ),
            "distribution": distribution,
        }

    @staticmethod
    def _forward_fraction(histogram: np.ndarray, edges: np.ndarray) -> float:
        """Return the binned fraction with a non-negative direction cosine."""
        total = int(histogram.sum())
        if not total:
            return 0.0
        centers = (edges[:-1] + edges[1:]) / 2.0
        return float(histogram[centers >= 0.0].sum() / total)

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> list[Path]:
        """Render one directly regenerable plot for each node task."""
        paths = []
        for key, task in summary["tasks"].items():
            if task["type"] == "classification":
                figure = plot_confusion_matrix(task["matrix"], task["class_names"])
            else:
                distribution = task["distribution"]
                figure = plot_histogram_with_boxplot(
                    {"Direction cosine": distribution},
                    distribution["histogram_edges"],
                    x_label="Truth/reconstruction direction cosine",
                    yscale="linear",
                )
            paths.extend(save_figure(figure, output_dir / f"node_{key}", formats))
        return paths


__all__ = ["NodeSummaryRecipe", "quality_cut_mask"]
