"""Batch reduction and rendering recipes for SPINE metric CSV files.

This module intentionally depends only on the lightweight scientific Python
stack. Plotting imports are deferred until rendering so JSON-only reports do
not require Matplotlib.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_SCHEMA_VERSION = "1.0.0"
DEFAULT_CHUNKSIZE = 100_000


def _plotting():
    """Load a non-interactive Matplotlib backend on demand."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _save_figure(fig: Any, output: Path, formats: Sequence[str]) -> None:
    """Save and close a figure in each requested graphical format."""
    plt = _plotting()
    for file_format in formats:
        if file_format != "json":
            fig.savefig(output.with_suffix(f".{file_format}"), bbox_inches="tight")
    plt.close(fig)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide arrays while returning zero for empty denominator bins."""
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _event_columns(columns: Iterable[str]) -> list[str]:
    """Return the strongest available event identity columns."""
    available = set(columns)
    run_columns = [name for name in ("run", "subrun", "event") if name in available]
    if len(run_columns) == 3:
        return run_columns
    return [name for name in ("file_index", "index") if name in available]


class _InputCounts:
    """Track event and source-file identities without retaining metric rows."""

    def __init__(self) -> None:
        self.events: set[tuple[Any, ...]] = set()
        self.files: set[tuple[str, Any]] = set()

    def update(self, path: Path, chunk: pd.DataFrame) -> None:
        """Add identities found in one CSV chunk."""
        columns = _event_columns(chunk.columns)
        if columns:
            values = chunk[columns].drop_duplicates().itertuples(index=False, name=None)
            self.events.update((str(path), *value) for value in values)
        if "file_index" in chunk:
            self.files.update((str(path), value) for value in chunk.file_index.unique())

    def as_dict(self, paths: Sequence[Path], row_count: int) -> dict[str, int]:
        """Build serializable input counts for a recipe summary."""
        return {
            "csv_shards": len(paths),
            "rows": row_count,
            "events": len(self.events),
            "data_files": len(self.files),
        }


class ReportRecipe(ABC):
    """Base interface shared by metric report recipes."""

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
    ) -> None:
        """Render plots using only values present in ``summary``."""


class SegmentConfusionRecipe(ReportRecipe):
    """Incrementally sum event-wise semantic confusion counts."""

    name = "segment_confusion"
    _column = re.compile(r"^count_(\d)(\d)$")

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        paths = list(csv_paths["source"])
        class_names = list(self.config.get("class_names", []))
        num_classes = len(class_names) or self.config.get("num_classes")
        matrix: np.ndarray | None = None
        counts = _InputCounts()
        row_count = 0

        for path in paths:
            for chunk in pd.read_csv(
                path, chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE)
            ):
                count_columns = {
                    match.groups(): column
                    for column in chunk.columns
                    if (match := self._column.match(column))
                }
                if not count_columns:
                    raise ValueError(f"No confusion count columns found in {path}.")
                inferred = max(max(int(i), int(j)) for i, j in count_columns) + 1
                size = int(num_classes or inferred)
                if inferred > size:
                    raise ValueError(
                        f"Configured {size} classes, but {path} contains {inferred}."
                    )
                if matrix is None:
                    matrix = np.zeros((size, size), dtype=np.int64)
                for (pred, label), column in count_columns.items():
                    matrix[int(pred), int(label)] += int(chunk[column].sum())
                counts.update(path, chunk)
                row_count += len(chunk)

        assert matrix is not None
        if not class_names:
            class_names = [str(index) for index in range(len(matrix))]
        if len(class_names) != len(matrix):
            raise ValueError("Must provide one class name per segmentation class.")

        support = matrix.sum(axis=0)
        predicted = matrix.sum(axis=1)
        diagonal = np.diag(matrix)
        return {
            "recipe": self.name,
            "inputs": counts.as_dict(paths, row_count),
            "class_names": class_names,
            "matrix": matrix.tolist(),
            "recall_matrix": _safe_ratio(matrix, support).tolist(),
            "precision_matrix": _safe_ratio(matrix, predicted[:, None]).tolist(),
            "per_class": {
                name: {
                    "support": int(support[index]),
                    "predicted": int(predicted[index]),
                    "recall": float(_safe_ratio(diagonal, support)[index]),
                    "precision": float(_safe_ratio(diagonal, predicted)[index]),
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
    ) -> None:
        plt = _plotting()
        matrix = np.asarray(summary["matrix"])
        normalized = np.asarray(summary["recall_matrix"])
        labels = summary["class_names"]
        fig, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for pred in range(len(matrix)):
            for truth in range(len(matrix)):
                value = normalized[pred, truth]
                axis.text(
                    truth,
                    pred,
                    f"{value:.3f}\n({matrix[pred, truth]:d})",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                )
        axis.set(
            xlabel="True class",
            ylabel="Predicted class",
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        fig.colorbar(image, ax=axis, label="Recall-normalized fraction")
        _save_figure(fig, output_dir / "segmentation_confusion", formats)


class PointProposalRecipe(ReportRecipe):
    """Stream bidirectional point distances into thresholds and histograms."""

    name = "point_proposal"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        thresholds = np.asarray(
            self.config.get("distance_thresholds", (1.0, 2.0, 5.0)),
            dtype=np.float64,
        )
        if thresholds.ndim != 1 or len(thresholds) == 0 or np.any(thresholds < 0):
            raise ValueError(
                "PPN distance thresholds must be a non-empty positive list."
            )
        thresholds.sort()
        distance_range = self.config.get("distance_range", [0.0, float(thresholds[-1])])
        bins = int(self.config.get("bins", 50))
        edges = np.linspace(
            float(distance_range[0]), float(distance_range[1]), bins + 1
        )
        scale = float(self.config.get("distance_scale", 1.0))
        directions = {}
        all_paths: list[Path] = []

        for source_key, label in (
            ("truth_to_reco", "efficiency"),
            ("reco_to_truth", "purity"),
        ):
            paths = list(csv_paths[source_key])
            all_paths.extend(paths)
            total = matched = correct_type = 0
            passing = np.zeros(len(thresholds), dtype=np.int64)
            histogram = np.zeros(bins, dtype=np.int64)
            counts = _InputCounts()
            row_count = 0
            for path in paths:
                for chunk in pd.read_csv(
                    path,
                    chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
                ):
                    if "dist" not in chunk:
                        raise ValueError(f"Missing `dist` column in {path}.")
                    distances = chunk.dist.to_numpy(dtype=np.float64) * scale
                    valid = np.isfinite(distances) & (distances >= 0.0)
                    total += len(distances)
                    matched += int(valid.sum())
                    passing += np.asarray(
                        [
                            np.count_nonzero(valid & (distances <= value))
                            for value in thresholds
                        ]
                    )
                    histogram += np.histogram(distances[valid], bins=edges)[0]
                    if {"shape", "closest_shape"} <= set(chunk.columns):
                        correct_type += int(
                            np.count_nonzero(
                                valid & (chunk["shape"] == chunk["closest_shape"])
                            )
                        )
                    counts.update(path, chunk)
                    row_count += len(chunk)
            directions[label] = {
                "inputs": counts.as_dict(paths, row_count),
                "total": total,
                "matched": matched,
                "threshold_fraction": {
                    str(float(value)): float(count / total) if total else 0.0
                    for value, count in zip(thresholds, passing)
                },
                "type_accuracy": float(correct_type / matched) if matched else 0.0,
                "histogram": histogram.tolist(),
            }

        return {
            "recipe": self.name,
            "inputs": {"csv_shards": len(all_paths)},
            "distance_thresholds": thresholds.tolist(),
            "distance_unit": self.config.get("distance_unit", "cm"),
            "histogram_edges": edges.tolist(),
            "directions": directions,
        }

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> None:
        plt = _plotting()
        thresholds = np.asarray(summary["distance_thresholds"])
        unit = summary["distance_unit"]
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
            _save_figure(fig, output_dir / f"ppn_{label}", formats)

        edges = np.asarray(summary["histogram_edges"])
        centers = (edges[1:] + edges[:-1]) / 2.0
        fig, axis = plt.subplots(figsize=(9, 6))
        for label, values in summary["directions"].items():
            axis.step(
                centers, values["histogram"], where="mid", label=label.capitalize()
            )
        axis.set(xlabel=f"Closest-point distance [{unit}]", ylabel="Points")
        axis.set_yscale("log")
        axis.grid(True)
        axis.legend()
        _save_figure(fig, output_dir / "ppn_resolution", formats)


class ClusterSummaryRecipe(ReportRecipe):
    """Stream clustering CSVs into scalar distribution summaries."""

    name = "cluster_summary"

    def reduce(self, csv_paths: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        metric_names = list(self.config.get("metric_names", ("ari", "eff", "pur")))
        bins = int(self.config.get("bins", 20))
        metric_ranges = {
            "ari": (-1.0, 1.0),
            "eff": (0.0, 1.0),
            "pur": (0.0, 1.0),
            **self.config.get("metric_ranges", {}),
        }
        levels = {}
        all_paths: list[Path] = []
        for level, paths_value in csv_paths.items():
            paths = list(paths_value)
            all_paths.extend(paths)
            accumulators = {
                metric: {
                    "count": 0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "histogram": np.zeros(bins, dtype=np.int64),
                    "edges": np.linspace(
                        *metric_ranges.get(metric, (0.0, 1.0)), bins + 1
                    ),
                }
                for metric in metric_names
            }
            counts = _InputCounts()
            row_count = 0
            for path in paths:
                for chunk in pd.read_csv(
                    path,
                    chunksize=self.config.get("chunksize", DEFAULT_CHUNKSIZE),
                ):
                    missing = set(metric_names) - set(chunk.columns)
                    if missing:
                        raise ValueError(
                            f"Missing clustering columns {sorted(missing)} in {path}."
                        )
                    for metric in metric_names:
                        values = chunk[metric].to_numpy(dtype=np.float64)
                        accumulator = accumulators[metric]
                        lower, upper = accumulator["edges"][[0, -1]]
                        values = values[
                            np.isfinite(values) & (values >= lower) & (values <= upper)
                        ]
                        accumulator["count"] += len(values)
                        accumulator["sum"] += float(values.sum())
                        accumulator["sum_sq"] += float(np.square(values).sum())
                        accumulator["histogram"] += np.histogram(
                            values, bins=accumulator["edges"]
                        )[0]
                    counts.update(path, chunk)
                    row_count += len(chunk)

            metrics = {}
            for metric, accumulator in accumulators.items():
                count = accumulator["count"]
                mean = accumulator["sum"] / count if count else 0.0
                variance = (
                    max(accumulator["sum_sq"] / count - mean**2, 0.0) if count else 0.0
                )
                metrics[metric] = {
                    "count": count,
                    "mean": mean,
                    "std": float(np.sqrt(variance)),
                    "histogram": accumulator["histogram"].tolist(),
                    "histogram_edges": accumulator["edges"].tolist(),
                }
            levels[level] = {
                "inputs": counts.as_dict(paths, row_count),
                "metrics": metrics,
            }

        return {
            "recipe": self.name,
            "inputs": {"csv_shards": len(all_paths)},
            "levels": levels,
        }

    def render(
        self,
        summary: Mapping[str, Any],
        output_dir: Path,
        formats: Sequence[str],
    ) -> None:
        plt = _plotting()
        levels = list(summary["levels"])
        metric_names = list(next(iter(summary["levels"].values()))["metrics"])
        positions = np.arange(len(levels), dtype=np.float64)
        width = 0.8 / len(metric_names)
        fig, axis = plt.subplots(figsize=(10, 6))
        for index, metric in enumerate(metric_names):
            values = [
                summary["levels"][level]["metrics"][metric]["mean"] for level in levels
            ]
            errors = [
                summary["levels"][level]["metrics"][metric]["std"] for level in levels
            ]
            axis.bar(
                positions + (index - (len(metric_names) - 1) / 2.0) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=metric.upper(),
            )
        axis.set(
            ylabel="Metric mean (error bar: standard deviation)",
            ylim=(0.0, 1.02),
            xticks=positions,
            xticklabels=levels,
        )
        axis.grid(True, axis="y")
        axis.legend()
        _save_figure(fig, output_dir / "clustering_summary", formats)


RECIPE_REGISTRY = {
    recipe.name: recipe
    for recipe in (SegmentConfusionRecipe, PointProposalRecipe, ClusterSummaryRecipe)
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 checksum of a file without loading it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _metadata(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Normalize configured provenance and fill deterministic checksums."""
    metadata = dict(config.get("metadata", {}))
    metadata["report_schema_version"] = REPORT_SCHEMA_VERSION
    metadata["report_config"] = str(config_path.resolve())
    metadata["report_config_sha256"] = _sha256(config_path)
    checkpoint = metadata.get("checkpoint")
    if isinstance(checkpoint, str):
        checkpoint = {"path": checkpoint}
    if isinstance(checkpoint, Mapping):
        checkpoint = dict(checkpoint)
        path = Path(str(checkpoint.get("path", ""))).expanduser()
        if path.is_file() and "sha256" not in checkpoint:
            checkpoint["sha256"] = _sha256(path)
        metadata["checkpoint"] = checkpoint
    return metadata


def _patterns(metric_config: Mapping[str, Any]) -> dict[str, str]:
    """Extract named input glob patterns from one recipe configuration."""
    if "source" in metric_config:
        return {"source": str(metric_config["source"])}
    for key in ("sources",):
        if key in metric_config:
            value = metric_config[key]
            if not isinstance(value, Mapping):
                raise TypeError(f"`{key}` must map input names to glob patterns.")
            return {str(name): str(pattern) for name, pattern in value.items()}
    point_patterns = {
        key: str(metric_config[key])
        for key in ("truth_to_reco", "reco_to_truth")
        if key in metric_config
    }
    if point_patterns:
        return point_patterns
    raise ValueError("Metric recipe must define `source`, `sources`, or PPN inputs.")


def build_report(
    config_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reduce configured CSV shards, render plots, and write ``summary.json``.

    Parameters
    ----------
    config_path : str or Path
        Report YAML configuration path.
    input_dir : str or Path
        Root directory beneath which recipe glob patterns are evaluated.
    output_dir : str or Path
        Destination directory for the JSON summary and plots.

    Returns
    -------
    dict
        The same serializable dictionary written to ``summary.json``.
    """
    import yaml

    config_path = Path(config_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, Mapping):
        raise TypeError("Report configuration must be a mapping.")
    metrics = config.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("Report configuration must contain a non-empty `metrics` map.")

    strict = bool(config.get("strict", True))
    formats = list(config.get("formats", ("png", "json")))
    supported_formats = {"json", "png", "pdf", "svg"}
    if unsupported := set(formats) - supported_formats:
        raise ValueError(f"Unsupported report formats: {sorted(unsupported)}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": _metadata(config, config_path),
        "inputs": {"directory": str(input_dir.resolve()), "csv_shards": 0},
        "metrics": {},
    }
    recipes: list[ReportRecipe] = []
    for key, raw_metric_config in metrics.items():
        if not isinstance(raw_metric_config, Mapping):
            raise TypeError(f"Metric `{key}` configuration must be a mapping.")
        metric_config = dict(raw_metric_config)
        recipe_name = metric_config.get("name")
        if recipe_name not in RECIPE_REGISTRY:
            raise ValueError(
                f"Unknown report recipe `{recipe_name}` for metric `{key}`."
            )
        discovered = {
            name: sorted(path for path in input_dir.glob(pattern) if path.is_file())
            for name, pattern in _patterns(metric_config).items()
        }
        missing = [name for name, paths in discovered.items() if not paths]
        if missing:
            message = f"Metric `{key}` found no CSV files for inputs: {missing}."
            if strict:
                raise FileNotFoundError(message)
            result["metrics"][key] = {
                "recipe": recipe_name,
                "status": "skipped",
                "reason": message,
            }
            continue

        recipe = RECIPE_REGISTRY[recipe_name](str(key), metric_config)
        summary = recipe.reduce(discovered)
        summary["sources"] = {
            name: [str(path.relative_to(input_dir)) for path in paths]
            for name, paths in discovered.items()
        }
        result["metrics"][key] = summary
        result["inputs"]["csv_shards"] += sum(
            len(paths) for paths in discovered.values()
        )
        recipes.append(recipe)

    event_counts = [
        source["inputs"]["events"]
        for metric in result["metrics"].values()
        for source in (
            metric.get("levels", {}).values()
            if "levels" in metric
            else (
                metric.get("directions", {}).values()
                if "directions" in metric
                else (metric,)
            )
        )
        if "inputs" in source and "events" in source["inputs"]
    ]
    result["inputs"]["events"] = max(event_counts, default=0)
    file_counts = [
        source["inputs"]["data_files"]
        for metric in result["metrics"].values()
        for source in (
            metric.get("levels", {}).values()
            if "levels" in metric
            else (
                metric.get("directions", {}).values()
                if "directions" in metric
                else (metric,)
            )
        )
        if "inputs" in source and "data_files" in source["inputs"]
    ]
    result["inputs"]["data_files"] = max(file_counts, default=0)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if any(file_format != "json" for file_format in formats):
        for recipe in recipes:
            recipe.render(result["metrics"][recipe.key], output_dir, formats)
    return result


__all__ = [
    "ClusterSummaryRecipe",
    "PointProposalRecipe",
    "REPORT_SCHEMA_VERSION",
    "ReportRecipe",
    "SegmentConfusionRecipe",
    "build_report",
]
