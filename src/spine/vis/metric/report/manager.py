"""Configuration, discovery, provenance and orchestration for reports.

The report manager is the filesystem-facing layer behind ``spine-report``. It
loads a standalone YAML configuration, discovers completed analyzer shards,
dispatches each metric to its recipe, and writes both figures and a cumulative
``summary.json``. Model and I/O-stack imports are intentionally absent so the
manager remains suitable for a lightweight CPU batch job.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import REPORT_SCHEMA_VERSION, ReportRecipe
from .cluster import ClusterSummaryRecipe
from .node import NodeSummaryRecipe
from .point import PointProposalRecipe
from .segment import SegmentConfusionRecipe

RECIPE_REGISTRY = {
    recipe.name: recipe
    for recipe in (
        SegmentConfusionRecipe,
        PointProposalRecipe,
        ClusterSummaryRecipe,
        NodeSummaryRecipe,
    )
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest without loading it into memory.

    Parameters
    ----------
    path : Path
        File to hash.
    block_size : int, default 1048576
        Number of bytes read per iteration.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _metadata(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Normalize configured provenance and fill deterministic checksums.

    Parameters
    ----------
    config : mapping
        Complete report configuration.
    config_path : Path
        Path from which the configuration was loaded.

    Returns
    -------
    dict
        User metadata augmented with schema version, resolved config path and
        config checksum. A local checkpoint receives a checksum when readable.
    """
    metadata = dict(config.get("metadata", {}))
    metadata["report_schema_version"] = REPORT_SCHEMA_VERSION
    metadata["report_config"] = str(config_path.resolve())
    metadata["report_config_sha256"] = _sha256(config_path)

    # A string is convenient in YAML; the JSON schema uses an extensible map.
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
    """Extract named input glob patterns from one recipe configuration.

    Recipes may use a single ``source``, a named ``sources`` mapping, or the
    two conventional PPN directions ``truth_to_reco`` and ``reco_to_truth``.

    Parameters
    ----------
    metric_config : mapping
        One metric entry from the report configuration.

    Returns
    -------
    dict
        Input names mapped to glob patterns.

    Raises
    ------
    TypeError
        If ``sources`` is present but is not a mapping.
    ValueError
        If the metric defines no recognized input form.
    """
    if "source" in metric_config:
        return {"source": str(metric_config["source"])}
    if "sources" in metric_config:
        value = metric_config["sources"]
        if not isinstance(value, Mapping):
            raise TypeError("`sources` must map input names to glob patterns.")
        return {str(name): str(pattern) for name, pattern in value.items()}

    point_patterns = {
        key: str(metric_config[key])
        for key in ("truth_to_reco", "reco_to_truth")
        if key in metric_config
    }
    if point_patterns:
        return point_patterns
    raise ValueError("Metric recipe must define `source`, `sources`, or PPN inputs.")


def _discover(
    metric_key: str,
    metric_config: Mapping[str, Any],
    input_dir: Path,
    *,
    strict: bool,
) -> tuple[dict[str, list[Path]], str | None]:
    """Resolve every configured input pattern beneath the input directory.

    Parameters
    ----------
    metric_key : str
        User-defined metric name used in diagnostics.
    metric_config : mapping
        Recipe configuration containing input patterns.
    input_dir : Path
        Root below which the patterns are evaluated.
    strict : bool
        Raise for unmatched inputs when ``True``; otherwise return a skip
        reason for the summary.

    Returns
    -------
    discovered : dict
        Sorted regular files grouped by input name.
    reason : str or None
        Human-readable skip reason in non-strict mode, otherwise ``None``.

    Raises
    ------
    FileNotFoundError
        If a required pattern matches no files in strict mode.
    """
    discovered = {
        name: sorted(path for path in input_dir.glob(pattern) if path.is_file())
        for name, pattern in _patterns(metric_config).items()
    }
    missing = [name for name, paths in discovered.items() if not paths]
    if not missing:
        return discovered, None

    message = f"Metric `{metric_key}` found no CSV files for inputs: {missing}."
    if strict:
        raise FileNotFoundError(message)
    return discovered, message


def _metric_sources(
    discovered: Mapping[str, list[Path]], input_dir: Path
) -> dict[str, list[str]]:
    """Express discovered source paths relative to the report input root.

    Parameters
    ----------
    discovered : mapping
        Absolute or input-root-relative paths grouped by source name.
    input_dir : Path
        Common report input root.

    Returns
    -------
    dict
        Source paths serialized relative to ``input_dir`` for portability.
    """
    return {
        name: [str(path.relative_to(input_dir)) for path in paths]
        for name, paths in discovered.items()
    }


def _nested_input_counts(metric: Mapping[str, Any], count_name: str) -> list[int]:
    """Collect one input count from the different recipe summary layouts.

    Parameters
    ----------
    metric : mapping
        Completed recipe summary.
    count_name : str
        Input statistic to collect, such as ``events`` or ``data_files``.

    Returns
    -------
    list of int
        Available counts across levels, directions or node sources.
    """
    if "levels" in metric:
        sources = metric["levels"].values()
    elif "directions" in metric:
        sources = metric["directions"].values()
    elif metric.get("recipe") == "node_summary":
        sources = metric["inputs"].values()
        return [source[count_name] for source in sources if count_name in source]
    else:
        sources = (metric,)
    return [
        source["inputs"][count_name]
        for source in sources
        if "inputs" in source and count_name in source["inputs"]
    ]


def _load_config(config_path: Path) -> Mapping[str, Any]:
    """Load and minimally validate a standalone report YAML file.

    Parameters
    ----------
    config_path : Path
        YAML configuration path.

    Returns
    -------
    mapping
        Parsed configuration containing a non-empty ``metrics`` mapping.

    Raises
    ------
    TypeError
        If the YAML root is not a mapping.
    ValueError
        If ``metrics`` is absent, empty or not a mapping.
    """
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, Mapping):
        raise TypeError("Report configuration must be a mapping.")
    metrics = config.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("Report configuration must contain a non-empty `metrics` map.")
    return config


def _refresh_input_counts(result: dict[str, Any]) -> None:
    """Update aggregate event and data-file counts from completed recipes.

    Counts can be repeated across recipes which consume the same events. The
    report therefore records the largest completed recipe count instead of
    summing and double-counting shared inputs.

    Parameters
    ----------
    result : dict
        Mutable top-level report summary.
    """
    event_counts = [
        value
        for metric in result["metrics"].values()
        for value in _nested_input_counts(metric, "events")
    ]
    file_counts = [
        value
        for metric in result["metrics"].values()
        for value in _nested_input_counts(metric, "data_files")
    ]
    result["inputs"]["events"] = max(event_counts, default=0)
    result["inputs"]["data_files"] = max(file_counts, default=0)


def _write_summary(result: Mapping[str, Any], path: Path) -> None:
    """Persist the current serializable report state as formatted JSON.

    Parameters
    ----------
    result : mapping
        Current report state. Non-finite JSON values are rejected.
    path : Path
        Destination ``summary.json`` path.
    """
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build_report(
    config_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reduce completed CSV shards and immediately write plots and JSON.

    Each recipe first creates a serializable dictionary, then renders directly
    from that dictionary. Figures are saved as soon as their metric reduction
    completes, so a long multi-recipe report exposes useful artifacts while
    later recipes are still running.

    Parameters
    ----------
    config_path : str or Path
        Report YAML configuration path.
    input_dir : str or Path
        Root directory beneath which recipe glob patterns are evaluated.
    output_dir : str or Path
        Destination directory for ``summary.json`` and generated plots.

    Returns
    -------
    dict
        The same serializable dictionary written to ``summary.json``.

    Raises
    ------
    FileNotFoundError
        If strict discovery is enabled and a configured source is absent.
    TypeError
        If the configuration or a metric entry has the wrong structure.
    ValueError
        If a recipe or output format is unsupported, or recipe validation
        fails.
    """
    config_path = Path(config_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config = _load_config(config_path)
    metrics = config["metrics"]
    strict = bool(config.get("strict", True))
    formats = list(config.get("formats", ("png", "json")))
    supported_formats = {"json", "png", "pdf", "svg"}
    if unsupported := set(formats) - supported_formats:
        raise ValueError(f"Unsupported report formats: {sorted(unsupported)}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": _metadata(config, config_path),
        "inputs": {
            "directory": str(input_dir.resolve()),
            "csv_shards": 0,
            "events": 0,
            "data_files": 0,
        },
        "metrics": {},
    }
    summary_path = output_dir / "summary.json"
    graphical_formats = [value for value in formats if value != "json"]
    for key, raw_metric_config in metrics.items():
        if not isinstance(raw_metric_config, Mapping):
            raise TypeError(f"Metric `{key}` configuration must be a mapping.")
        metric_config = dict(raw_metric_config)
        recipe_name = metric_config.get("name")
        if recipe_name not in RECIPE_REGISTRY:
            raise ValueError(
                f"Unknown report recipe `{recipe_name}` for metric `{key}`."
            )

        # Discovery occurs immediately before reduction, after the scheduler
        # has declared the producing inference jobs complete.
        discovered, missing_reason = _discover(
            str(key),
            metric_config,
            input_dir,
            strict=strict,
        )
        if missing_reason:
            result["metrics"][key] = {
                "recipe": recipe_name,
                "status": "skipped",
                "reason": missing_reason,
            }
            _write_summary(result, summary_path)
            continue

        # A recipe summary is the sole source for both JSON and plots.
        recipe: ReportRecipe = RECIPE_REGISTRY[recipe_name](str(key), metric_config)
        summary = recipe.reduce(discovered)
        summary["sources"] = _metric_sources(discovered, input_dir)
        summary["artifacts"] = []
        result["metrics"][key] = summary
        result["inputs"]["csv_shards"] += sum(
            len(paths) for paths in discovered.values()
        )

        # Render now rather than waiting for all remaining recipes to finish.
        if graphical_formats:
            artifacts = recipe.render(summary, output_dir, graphical_formats)
            summary["artifacts"] = [path.name for path in artifacts]

        # Keep records beside plots throughout the run. If a later recipe
        # fails, all previously completed metrics remain inspectable.
        _refresh_input_counts(result)
        _write_summary(result, summary_path)

    _refresh_input_counts(result)
    _write_summary(result, summary_path)
    return result


__all__ = ["RECIPE_REGISTRY", "build_report"]
