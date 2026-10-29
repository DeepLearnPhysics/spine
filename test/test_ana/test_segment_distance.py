"""Distance summaries preserve label alignment and physical coordinates."""

import numpy as np
import pytest

from spine.ana.metric.segment import SegmentAna
from spine.data import Meta, TensorData


def run_distance(monkeypatch, vertices, **kwargs):
    """Evaluate four voxels with a non-unit, anisotropic voxel size."""
    rows = {}
    monkeypatch.setattr(SegmentAna, "initialize_writer", lambda *args: None)
    monkeypatch.setattr(
        SegmentAna,
        "append",
        lambda self, name, **row: rows.setdefault(name, []).append(row),
    )
    meta = Meta(
        lower=np.array([10.0, 20.0, 30.0]),
        upper=np.array([30.0, 60.0, 90.0]),
        size=np.array([2.0, 4.0, 6.0]),
        count=np.array([10, 10, 10]),
    )
    labels = TensorData(
        coords=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]]),
        features=np.array([[0], [1], [1], [0]]),
        meta=meta,
    )
    data = {
        "seg_label": labels,
        "segmentation": np.eye(2)[[0, 0, 1, 1]],
        "vertices": TensorData(
            coords=np.asarray(vertices).reshape(-1, 3),
            features=np.zeros((len(vertices), 1)),
            meta=meta,
        ),
    }
    data.update(kwargs.pop("data", {}))
    ana = SegmentAna(
        vertex_key="vertices",
        distance_bins=[0, 2, 5],
        num_classes=kwargs.pop("num_classes", 2),
        **kwargs,
    )
    ana.process(data)
    return rows


def test_physical_bins_and_nearest_vertex(monkeypatch):
    """Exact edges go to the next bin; nearest of multiple vertices is used."""
    rows = run_distance(monkeypatch, [[0.5, 0.5, 0.5], [4.5, 0.5, 0.5]])
    bins = rows["distance"]
    assert bins[0]["count_00"] == 0
    assert bins[1]["count_00"] == 1
    assert bins[1]["count_10"] == 1
    assert bins[2]["count_01"] == bins[2]["count_11"] == 1
    assert sum(v for k, v in bins[3].items() if k.startswith("count_")) == 0
    for key, count in rows["summary"][0].items():
        assert sum(b[key] for b in bins) == count


def test_no_vertex_and_overflow(monkeypatch):
    """No-vertex counts stay separate from overflow and overall counts."""
    rows = run_distance(monkeypatch, [])
    assert {
        k: v for k, v in rows["distance"][0].items() if k.startswith("count_")
    } == rows["summary"][0]
    assert all(
        sum(v for k, v in row.items() if k.startswith("count_")) == 0
        for row in rows["distance"][1:]
    )
    rows = run_distance(monkeypatch, [[0.5, 0.5, 0.5]])
    assert rows["distance"][-1]["count_10"] == 1


def test_ghost_uses_full_label_coordinates(monkeypatch):
    """Rejected voxels retain the coordinates of the original label domain."""
    rows = run_distance(
        monkeypatch,
        [[0.5, 0.5, 0.5]],
        ghost=True,
        num_classes=5,
        data={
            "segmentation": np.eye(5)[[0, 1]],
            "ghost": np.eye(2)[[0, 1, 0, 1]],
        },
    )
    assert rows["distance"][2]["count_51"] == 1
    assert rows["distance"][3]["count_50"] == 1


@pytest.mark.parametrize("edges", [[], [1, 2], [0, 0], [0, -1], [0, float("inf")]])
def test_invalid_edges(monkeypatch, edges):
    """Invalid bins fail during analyzer construction."""
    monkeypatch.setattr(SegmentAna, "initialize_writer", lambda *args: None)
    with pytest.raises(ValueError, match="distance_bins"):
        SegmentAna(vertex_key="vertices", distance_bins=edges)


@pytest.mark.parametrize("vertices", [[], [[0.5, 0.5, 0.5]]])
def test_empty_evaluated_image(monkeypatch, vertices):
    """Empty images preserve a complete zero-count schema."""
    meta = Meta(
        lower=np.zeros(3),
        upper=np.ones(3),
        size=np.ones(3),
        count=np.ones(3, dtype=int),
    )
    rows = run_distance(
        monkeypatch,
        vertices,
        data={
            "seg_label": TensorData(
                coords=np.empty((0, 3)), features=np.empty((0, 1)), meta=meta
            ),
            "segmentation": np.empty((0, 2)),
        },
    )
    assert all(
        value == 0
        for row in rows["distance"]
        for key, value in row.items()
        if key.startswith("count_")
    )
