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
