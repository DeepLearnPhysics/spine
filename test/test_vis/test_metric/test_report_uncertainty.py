"""Classification denominators and whole-event bootstrap regression tests."""

import json

import numpy as np
import pandas as pd
import pytest

from spine.vis.metric.report.classification import (
    classification_metrics,
    resolve_class_groups,
)
from spine.vis.metric.report.segment import SegmentConfusionRecipe
from spine.vis.metric.report.segment_distance import SegmentDistanceRecipe
from spine.vis.metric.report.uncertainty import EventBootstrap


def groups(size=2):
    """Return canonical shape groups for synthetic confusion matrices."""
    return resolve_class_groups({}, kind="shape", default_ids=range(size))


def row(matrix, event, **extra):
    """Encode one event confusion matrix using analyzer column conventions."""
    return {
        "index": event,
        **{f"count_{i}{j}": int(v) for (i, j), v in np.ndenumerate(matrix)},
        **extra,
    }


def test_metrics_mapping_selection_and_macro():
    """Filtering retains false positives/negatives and pooling counts matches."""
    matrix = np.array([[8, 2, 3], [1, 4, 0], [6, 0, 0]])
    selected = resolve_class_groups(
        {"classes": [0]}, kind="shape", default_ids=range(3)
    )
    result = classification_metrics(matrix, selected)
    shower = result["per_class"]["Shower"]
    assert shower == {
        "support": 15,
        "predicted": 13,
        "recall": 8 / 15,
        "precision": 8 / 13,
        "f1": 16 / 28,
    }
    assert result["accuracy"] == 8 / 15
    assert result["macro_f1"] == 16 / 28
    mapped = resolve_class_groups(
        {"class_mapping": {"Combined": [0, 1], "Other": [2]}},
        kind="shape",
        default_ids=range(3),
    )
    pooled = classification_metrics(matrix, mapped)
    assert pooled["per_class"]["Combined"]["f1"] == 30 / 39
    assert pooled["macro_f1"] == 15 / 39


def test_undefined_missed_and_spurious_classes():
    """Absent classes are null; missed/spurious classes contribute zero F1."""
    matrix = np.zeros((4, 4), int)
    matrix[0, 0] = 2
    matrix[1, 2] = 3
    result = classification_metrics(matrix, groups(4))
    assert result["macro_f1"] == 1 / 3
    assert result["macro_f1_classes"] == ["Shower", "Track", "Michel"]
    assert result["per_class"]["Track"]["recall"] is None
    assert result["per_class"]["Michel"]["precision"] is None
    assert result["per_class"]["Delta"]["f1"] is None
    empty = classification_metrics(np.zeros((2, 2), int), groups())
    assert empty["macro_f1"] is empty["accuracy"] is None
    json.dumps(empty, allow_nan=False)


def test_event_bootstrap_matches_explicit_resampling(tmp_path):
    """All bins share draws, and chunking never splits an event's sample unit."""
    matrices = np.array([[[9, 0], [1, 0]], [[0, 0], [1, 0]], [[2, 0], [2, 0]]])
    frame = pd.DataFrame(
        [
            row(matrix * scale, event, bin_index=index)
            for event, matrix in enumerate(matrices)
            for index, scale in ((0, 1), (1, 2))
        ]
    )
    config = {"replicates": 101, "seed": 42, "confidence": 0.8}
    collector = EventBootstrap(config)
    for start in range(len(frame)):
        collector.update(
            tmp_path / "a.csv", frame.iloc[start : start + 1], 2, distance=True
        )
    metadata, intervals = collector.summarize(groups(), [0, 1])
    assert metadata["events"] == 3
    assert intervals[0] == intervals[1]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(101):
        total = matrices[rng.choice(3, size=3, replace=True)].sum(axis=0)
        samples.append(total[0, 0] / total.sum())
    actual = intervals[0]["per_class"]["Shower"]["recall"]
    assert [actual["lower"], actual["upper"]] == pytest.approx(
        np.quantile(samples, [0.1, 0.9])
    )
    assert actual["valid_replicates"] == 101
    whole = EventBootstrap(config)
    whole.update(tmp_path / "a.csv", frame, 2, distance=True)
    assert whole.summarize(groups(), [0, 1]) == (metadata, intervals)
    # Equal event IDs in independent shards are distinct sampling units.
    whole.update(tmp_path / "b.csv", frame, 2, distance=True)
    assert whole.summarize(groups(), [0, 1])[0]["events"] == 6


def test_sparse_intervals_are_null_and_count_undefined_draws(tmp_path):
    """Do not imply measured variation from a class supported by one event."""
    frame = pd.DataFrame([row(np.diag([10, 0]), 0), row(np.diag([0, 20]), 1)])
    collector = EventBootstrap({"replicates": 100, "seed": 5})
    collector.update(tmp_path / "a.csv", frame, 2)
    _, intervals = collector.summarize(groups(), [0])
    interval = intervals[0]["per_class"]["Shower"]["recall"]
    assert interval["contributing_events"] == 1
    assert 0 < interval["valid_replicates"] < 100
    assert interval["lower"] is interval["upper"] is None
    assert intervals[0]["accuracy"]["lower"] == 1
    json.dumps(intervals, allow_nan=False)


@pytest.mark.parametrize(
    "config",
    [
        None,
        True,
        {"replicates": 1},
        {"replicates": 3.5},
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": float("nan")},
        {"seed": -1},
        {"seed": True},
        {"unknown": 3},
    ],
)
def test_invalid_bootstrap_options(config):
    """Invalid options fail explicitly rather than silently disabling intervals."""
    with pytest.raises(ValueError):
        EventBootstrap(config)


def test_bootstrap_requires_event_identity(tmp_path):
    """Do not silently resample rows when event identities are unavailable."""
    collector = EventBootstrap({})
    for frame in (
        pd.DataFrame({"count_00": [1]}),
        pd.DataFrame({"index": [None], "count_00": [1]}),
    ):
        with pytest.raises(ValueError, match="event identity"):
            collector.update(tmp_path / "a.csv", frame, 1)


def test_overall_and_distance_share_metrics_and_intervals(tmp_path):
    """Both report routes use identical definitions, samples and JSON values."""
    matrices = [np.array([[8, 2], [1, 3]]), np.array([[2, 5], [3, 6]])]
    overall_path, distance_path = tmp_path / "overall.csv", tmp_path / "distance.csv"
    pd.DataFrame([row(matrix, i) for i, matrix in enumerate(matrices)]).to_csv(
        overall_path, index=False
    )
    pd.DataFrame(
        [
            row(
                matrix if b == 0 else np.zeros((2, 2), int),
                i,
                bin_index=b,
                distance_min=0 if b == 0 else -1,
                distance_max=-1,
                num_vertices=1,
                vertex_key="vertices",
            )
            for i, matrix in enumerate(matrices)
            for b in (-1, 0)
        ]
    ).to_csv(distance_path, index=False)
    config = {
        "bootstrap": {"replicates": 30, "seed": 12},
        "chunksize": 1,
        "classes": [0],
    }
    overall_recipe = SegmentConfusionRecipe("overall", config)
    distance_recipe = SegmentDistanceRecipe("distance", config)
    overall = overall_recipe.reduce({"source": [overall_path]})
    distance = distance_recipe.reduce({"source": [distance_path]})
    metrics = overall["metrics"]
    assert all(distance["bins"][0][key] == value for key, value in metrics.items())
    assert overall["bootstrap"] == distance["bootstrap"]
    assert (
        overall["per_class"]["Shower"]["f1_overall"]
        == metrics["per_class"]["Shower"]["f1"]
    )
    assert distance["no_vertex"]["macro_f1"] is None
    json.dumps([overall, distance], allow_nan=False)
    assert all(
        path.exists() for path in overall_recipe.render(overall, tmp_path, ["png"])
    )
    assert all(
        path.exists() for path in distance_recipe.render(distance, tmp_path, ["png"])
    )
    plain = SegmentConfusionRecipe("plain", {}).reduce({"source": [overall_path]})
    assert "bootstrap" not in plain
    assert "confidence_intervals" not in plain["metrics"]


def test_single_event_empty_bins_and_invalid_counts(tmp_path):
    """Sparse inputs serialize safely and malformed counts never enter draws."""
    collector = EventBootstrap({"replicates": 10})
    frame = pd.DataFrame([row(np.diag([5, 0]), 0, bin_index=-1)])
    collector.update(tmp_path / "a.csv", frame, 2, distance=True)
    _, intervals = collector.summarize(groups(), [-1, 0])
    assert intervals[-1]["accuracy"]["lower"] is None
    assert intervals[-1]["accuracy"]["valid_replicates"] == 10
    assert intervals[0]["accuracy"]["valid_replicates"] == 0
    assert intervals[0]["macro_f1"]["lower"] is None
    json.dumps(intervals, allow_nan=False)
    for bad in (-1, 0.5, np.nan, np.inf):
        invalid = frame.astype({"count_00": float})
        invalid.loc[0, "count_00"] = bad
        with pytest.raises(ValueError, match="nonnegative integers"):
            collector.update(tmp_path / "a.csv", invalid, 2)
    with pytest.raises(ValueError, match="at least one event"):
        EventBootstrap({}).summarize(groups(), [0])
