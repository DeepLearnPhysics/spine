"""Integration tests for the optional distance report route."""

import json

import pandas as pd
import pytest

from spine.vis.metric.report.manager import RECIPE_REGISTRY
from spine.vis.metric.report.segment_distance import SegmentDistanceRecipe


def write_shard(path, multiplier=1):
    """Write two events, one without vertices, with complete bin metadata."""
    rows = []
    for event in range(2):
        for index, lower, upper in [(-1, -1, -1), (0, 0, 2), (1, 2, -1)]:
            active = (event == 0 and index == 0) or (event == 1 and index == -1)
            rows.append(
                dict(
                    index=event,
                    file_index=0,
                    vertex_key="vertices",
                    bin_index=index,
                    distance_min=lower,
                    distance_max=upper,
                    num_vertices=int(event == 0),
                    count_00=2 * multiplier if active else 0,
                    count_01=multiplier if active else 0,
                    count_10=0,
                    count_11=multiplier if active else 0,
                )
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_reduce_render_and_no_vertex(tmp_path):
    """Aggregate counts before division, retain null bins, render all outputs."""
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for i, path in enumerate(paths):
        write_shard(path, i + 1)
    assert RECIPE_REGISTRY["segment_distance"] is SegmentDistanceRecipe
    recipe = SegmentDistanceRecipe("vertex", {"chunksize": 2})
    result = recipe.reduce({"source": paths})
    assert result["bins"][0]["accuracy"] == 0.75
    assert result["bins"][0]["voxel_count"] == 12
    assert result["bins"][0]["per_class"]["Track"]["recall"] == 0.5
    assert result["bins"][1]["accuracy"] is None
    assert result["no_vertex"]["event_count"] == 2
    assert result["no_vertex"]["voxel_count"] == 12
    json.dumps(result, allow_nan=False)
    assert all(p.exists() for p in recipe.render(result, tmp_path, ["png"]))
    selected = SegmentDistanceRecipe("track", {"classes": ["track"]}).reduce(
        {"source": paths}
    )
    assert selected["bins"][0]["accuracy"] == 0.5


def test_reject_incompatible_bins(tmp_path):
    """Different edge definitions must never silently pool shards."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_shard(a)
    write_shard(b)
    frame = pd.read_csv(b)
    frame.loc[frame.bin_index == 0, "distance_max"] = 3
    frame.to_csv(b, index=False)
    with pytest.raises(ValueError, match="edges"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [a, b]})


def test_report_manager_route(tmp_path):
    """The public report entry point discovers and renders distance CSVs."""
    from spine.vis.metric.report import build_report

    raw = tmp_path / "raw"
    raw.mkdir()
    write_shard(raw / "segment_eval_distance.csv")
    config = tmp_path / "report.yaml"
    config.write_text(
        'formats: [json, png]\nmetrics:\n  vertices:\n    provider: segment_distance\n    source: "**/*segment_eval_distance.csv"\n'
    )
    build_report(config, raw, tmp_path / "out")
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "vertices_distance.png").exists()


@pytest.mark.parametrize("schema", ["missing", "incomplete", "different"])
def test_reject_incompatible_count_schemas(tmp_path, schema):
    """Shards must contain a complete matrix with consistent class dimensions."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_shard(a)
    write_shard(b)
    frame = pd.read_csv(b)
    if schema == "missing":
        frame = frame.drop(columns=[c for c in frame if c.startswith("count_")])
    elif schema == "incomplete":
        frame = frame.drop(columns="count_10")
    else:
        frame = frame.drop(columns=["count_01", "count_10", "count_11"])
    frame.to_csv(b, index=False)
    with pytest.raises(ValueError, match="confusion schema"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [a, b]})


def test_reject_configured_class_count_mismatch(tmp_path):
    """An explicit class count must agree with the CSV matrix dimensions."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    with pytest.raises(ValueError, match="num_classes differs"):
        SegmentDistanceRecipe("vertex", {"num_classes": 3}).reduce({"source": [path]})


@pytest.mark.parametrize("mixed_within_shard", [False, True])
def test_reject_different_vertex_products(tmp_path, mixed_within_shard):
    """Distances to different point collections cannot be pooled."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_shard(a)
    write_shard(b)
    frame = pd.read_csv(b)
    if mixed_within_shard:
        frame.loc[0, "vertex_key"] = "other_vertices"
    else:
        frame["vertex_key"] = "other_vertices"
    frame.to_csv(b, index=False)
    with pytest.raises(ValueError, match="different vertex products"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [a, b]})


@pytest.mark.parametrize("invalid", [-1, 0.5, float("nan"), float("inf")])
def test_reject_invalid_confusion_counts(tmp_path, invalid):
    """Negative, fractional and nonfinite counts are not valid statistics."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    frame = pd.read_csv(path).astype({"count_00": float})
    frame.loc[0, "count_00"] = invalid
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonnegative integers"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [path]})


@pytest.mark.parametrize("invalid", [-2, 0.5])
def test_reject_invalid_bin_indices(tmp_path, invalid):
    """Only integral bins starting at the reserved no-vertex index are valid."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    frame = pd.read_csv(path).astype({"bin_index": float})
    frame.loc[frame.bin_index == 0, "bin_index"] = invalid
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid distance bin index"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [path]})


@pytest.mark.parametrize("missing", [-1, 0])
def test_reject_missing_bins(tmp_path, missing):
    """No-vertex and contiguous distance records must be present."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    frame = pd.read_csv(path)
    frame.loc[frame.bin_index != missing].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing distance bins"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [path]})


def test_reject_empty_shard_list():
    """An empty input set cannot define distance-bin metrics."""
    with pytest.raises(ValueError, match="Missing distance bins"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": []})


@pytest.mark.parametrize(
    ("index", "column", "value", "message"),
    [
        (-1, "distance_min", 0, "no-vertex or first"),
        (-1, "distance_max", 0, "no-vertex or first"),
        (0, "distance_min", 1, "no-vertex or first"),
        (1, "distance_min", float("inf"), "contiguous and end in overflow"),
        (1, "distance_max", 10, "contiguous and end in overflow"),
        (0, "distance_max", 0, "contiguous and end in overflow"),
        (0, "distance_max", 3, "contiguous and end in overflow"),
    ],
)
def test_reject_invalid_interval_sequence(tmp_path, index, column, value, message):
    """Validate sentinel bounds, increasing intervals, adjacency and overflow."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    frame = pd.read_csv(path).astype({column: float})
    frame.loc[frame.bin_index == index, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [path]})


def test_reject_inconsistent_bounds_within_chunk(tmp_path):
    """One bin cannot have event-dependent edges within a single CSV chunk."""
    path = tmp_path / "counts.csv"
    write_shard(path)
    frame = pd.read_csv(path)
    frame.loc[(frame.bin_index == 0) & (frame["index"] == 1), "distance_max"] = 3
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Inconsistent distance bin edges"):
        SegmentDistanceRecipe("vertex", {}).reduce({"source": [path]})
