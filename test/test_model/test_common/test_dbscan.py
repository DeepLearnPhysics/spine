"""Tests for dense DBSCAN fragment clustering."""

import numpy as np
import pytest
import torch

from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.data import TensorBatch
from spine.model.common.dbscan import DBSCAN


def test_dbscan_preserves_indices_across_interleaved_shapes():
    """Cluster-local indices map back to the original voxel ordering."""
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 10.0, 0.0, 0.0, 1.0],
                [0.0, 0.5, 0.0, 0.0, 1.0],
            ]
        ),
        counts=torch.tensor([3]),
    )
    segmentation = TensorBatch(
        torch.tensor([SHOWR_SHP, TRACK_SHP, SHOWR_SHP]),
        counts=torch.tensor([3]),
    )
    clusterer = DBSCAN(
        eps=1.0,
        min_samples=1,
        min_size=1,
        shapes=(SHOWR_SHP, TRACK_SHP),
        break_shapes=(),
    )

    clusters, shapes = clusterer(data, segmentation)

    assert [set(cluster.tolist()) for cluster in clusters.index_list] == [
        {0, 2},
        {1},
    ]
    assert np.array_equal(shapes.numpy_tensor(), np.array([SHOWR_SHP, TRACK_SHP]))


def test_dbscan_expands_scalar_shape_parameters():
    """Scalar clustering settings apply independently to every shape."""
    clusterer = DBSCAN(
        eps=2.0,
        min_samples=2,
        min_size=3,
        metric="euclidean",
        shapes=(SHOWR_SHP, TRACK_SHP),
        break_shapes=(),
    )

    assert clusterer.eps == [2.0, 2.0]
    assert clusterer.min_samples == [2, 2]
    assert clusterer.min_size == [3, 3]
    assert clusterer.metric == ["euclidean", "euclidean"]


def test_dbscan_rejects_inconsistent_shape_parameters():
    """Per-shape settings must match the configured number of shapes."""
    with pytest.raises(ValueError, match="number of `eps`"):
        DBSCAN(
            eps=(1.0,),
            shapes=(SHOWR_SHP, TRACK_SHP),
            break_shapes=(),
        )
