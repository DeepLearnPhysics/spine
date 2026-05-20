"""Tests for GNN network utility functions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spine.utils.gnn.network import inter_cluster_distance

GRAPH_BASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "spine"
    / "model"
    / "layer"
    / "gnn"
    / "graph"
    / "base.py"
)


def load_graph_base():
    """Load GraphBase without importing the full GNN package."""
    spec = importlib.util.spec_from_file_location("graph_base", GRAPH_BASE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphBase


def test_recursive_distance_algorithm_warns_as_legacy_alias():
    """The recursive distance algorithm name should be visibly legacy."""
    graph_base = load_graph_base()

    class TestGraph(graph_base):
        name = "test"

    # The warning lives in GraphBase, not a specific concrete graph type.
    with pytest.warns(FutureWarning, match="does not perform recursive search"):
        graph = TestGraph(dist_algorithm="recursive")

    assert graph.dist_iterative is True
    assert graph.dist_legacy is True


def test_inter_cluster_distance_can_use_legacy_iterative_closest_pair():
    """Legacy distance mode should preserve historical iterative CPA output."""
    x1 = np.array(
        [
            [0.58366364, -1.8748202, 0.9472971],
            [-0.24740814, 0.6954392, 1.1409228],
            [0.22428122, -0.5900606, 1.20232],
        ],
        dtype=np.float32,
    )
    x2 = np.array(
        [
            [1.3192177, 0.69287896, 1.1638298],
            [-0.6025194, -0.69706947, 2.202115],
            [0.1937491, 0.1192039, 1.1976705],
            [0.3246087, -0.36247766, 1.2971592],
        ],
        dtype=np.float32,
    )
    voxels = np.vstack([x1, x2])
    clusts = [
        np.arange(len(x1), dtype=np.int64),
        np.arange(len(x1), len(x1) + len(x2), dtype=np.int64),
    ]
    counts = np.array([len(clusts)], dtype=np.int64)

    fixed_dist, fixed_index = inter_cluster_distance(
        voxels,
        clusts,
        counts,
        iterative=True,
        return_index=True,
        use_legacy_distance=False,
    )
    legacy_dist, legacy_index = inter_cluster_distance(
        voxels,
        clusts,
        counts,
        iterative=True,
        return_index=True,
        use_legacy_distance=True,
    )

    assert np.isclose(fixed_dist[0, 1], 0.2661843)
    assert fixed_index[0, 1] == 11
    assert np.isclose(legacy_dist[0, 1], 0.7099366)
    assert legacy_index[0, 1] == 10
