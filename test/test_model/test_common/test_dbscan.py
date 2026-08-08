"""Tests for dense DBSCAN fragment clustering."""

import numpy as np
import pytest
import torch

from spine.constants import DELTA_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import TensorBatch, TensorSchema
from spine.model.common.dbscan import DBSCAN


def make_data(shapes):
    """Build a canonical sparse batch and aligned semantic predictions."""
    rows = torch.zeros((len(shapes), 5), dtype=torch.float32)
    rows[:, 1] = torch.arange(len(shapes))
    rows[:, 4] = 1.0
    data = TensorBatch(
        rows,
        counts=[len(shapes)],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    segmentation = TensorBatch(torch.tensor(shapes), counts=[len(shapes)])
    return data, segmentation


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
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
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

    clusterer = DBSCAN(
        eps=(1.0, 2.0),
        min_samples=(1, 2),
        min_size=(3, 4),
        metric=("euclidean", "chebyshev"),
        shapes=(SHOWR_SHP, TRACK_SHP),
        break_shapes=(),
    )
    assert clusterer.eps == [1.0, 2.0]
    assert clusterer.metric == ["euclidean", "chebyshev"]


def test_dbscan_rejects_inconsistent_shape_parameters():
    """Per-shape settings must match the configured number of shapes."""
    with pytest.raises(ValueError, match="number of `eps`"):
        DBSCAN(
            eps=(1.0,),
            shapes=(SHOWR_SHP, TRACK_SHP),
            break_shapes=(),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shapes": SHOWR_SHP}, "classes should be provided as a sequence"),
        ({"break_shapes": TRACK_SHP}, "to break should be provided as a sequence"),
        ({"break_shapes": (TRACK_SHP,)}, "must provide a PPN predictor"),
    ],
)
def test_dbscan_validates_shape_and_point_configuration(kwargs, message):
    """Malformed shape collections and missing point sources fail early."""
    with pytest.raises(ValueError, match=message):
        DBSCAN(**kwargs)


def test_dbscan_label_points_break_tracks_and_remove_delta_voxels():
    """Truth endpoints drive point breaking while added delta voxels are removed."""
    data, segmentation = make_data([TRACK_SHP, DELTA_SHP, TRACK_SHP])
    clusterer = DBSCAN(
        shapes=(TRACK_SHP,),
        break_shapes=(TRACK_SHP,),
        use_label_break_points=True,
        track_include_delta=True,
        min_size=1,
    )
    seen = {}

    def break_track(voxels, points):
        seen["voxels"] = voxels
        seen["points"] = points
        return np.zeros(len(voxels), dtype=np.int64)

    clusterer.clusterers[0] = break_track
    coord_rows = torch.tensor([[0, 0, 0, 2, 0, 0, TRACK_SHP]], dtype=torch.float32)
    coord_label = TensorBatch(
        coord_rows,
        counts=[1],
        coord_cols=np.arange(6),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"shape": (0,)},
        ),
    )

    clusters, shapes = clusterer(data, segmentation, coord_label=coord_label)

    assert seen["voxels"].shape == (3, 3)
    assert seen["points"].shape == (2, 3)
    assert clusters.index_list[0].tolist() == [0, 2]
    assert shapes.numpy_tensor().tolist() == [TRACK_SHP]


def test_dbscan_label_breaking_requires_coordinate_labels():
    """Truth-driven point breaking fails clearly without point labels."""
    data, segmentation = make_data([TRACK_SHP])
    clusterer = DBSCAN(
        shapes=(TRACK_SHP,),
        break_shapes=(TRACK_SHP,),
        use_label_break_points=True,
    )

    with pytest.raises(ValueError, match="must provide them"):
        clusterer(data, segmentation)


def test_dbscan_nontrack_break_shape_uses_masked_method():
    """Non-track point splitting always uses the masked DBSCAN strategy."""
    clusterer = DBSCAN(
        shapes=(SHOWR_SHP,),
        break_shapes=(SHOWR_SHP,),
        ppn_predictor={},
    )

    assert clusterer.clusterers[0].method == "masked_dbscan"


def test_dbscan_uses_predicted_points_and_filters_delta_points():
    """The PPN path passes only non-delta points to point-aware clustering."""
    data, segmentation = make_data([TRACK_SHP, TRACK_SHP])
    clusterer = DBSCAN(
        shapes=(TRACK_SHP,),
        break_shapes=(TRACK_SHP,),
        ppn_predictor={},
        min_size=1,
    )
    points = TensorBatch(
        torch.tensor(
            [
                [0, 0, 0, 0, TRACK_SHP],
                [0, 1, 0, 0, DELTA_SHP],
            ],
            dtype=torch.float32,
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
        schema=TensorSchema(
            coordinate_groups={"points": (0, 1, 2)},
            feature_fields={"shape": (0,)},
        ),
    )
    clusterer.ppn_predictor = lambda **_result: points
    seen = {}

    def break_track(voxels, break_points):
        seen["points"] = break_points
        return np.arange(len(voxels), dtype=np.int64)

    clusterer.clusterers[0] = break_track
    clusters, _ = clusterer(data, segmentation, ppn_points=points)

    assert seen["points"].shape == (1, 3)
    assert len(clusters.index_list) == 2


def test_dbscan_returns_typed_empty_outputs_and_applies_minimum_size():
    """Empty and undersized classes produce stable empty batch products."""
    data, segmentation = make_data([SHOWR_SHP, SHOWR_SHP])
    clusterer = DBSCAN(
        shapes=(TRACK_SHP,),
        break_shapes=(),
        min_size=2,
    )
    clusters, shapes = clusterer(data, segmentation)
    assert len(clusters.index_list) == 0
    assert clusters.counts.tolist() == [0]
    assert shapes.counts.tolist() == [0]

    segmentation = TensorBatch(torch.tensor([TRACK_SHP, TRACK_SHP]), counts=[2])
    clusterer.clusterers[0] = lambda _voxels, _points: np.array([0, 1])
    clusters, _ = clusterer(data, segmentation)
    assert len(clusters.index_list) == 0
