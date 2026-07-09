from types import SimpleNamespace

import numpy as np
import torch

import spine.model.full_chain as full_chain_mod
from spine.constants import (
    CLUST_COL,
    GHOST_SHP,
    GROUP_COL,
    PRGRP_COL,
    SHAPE_COL,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.data import IndexBatch, TensorBatch
from spine.model.full_chain import FullChain, FullChainLoss


class RecordingLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "loss": torch.tensor(0.0),
            "accuracy": 1.0,
        }


def test_full_chain_loss_uses_orig_index_to_align_cached_segmentation_labels():
    """Cached deghosted segmentation inputs should be aligned via orig_index."""
    chain = {
        "deghosting": None,
        "charge_rescaling": None,
        "segmentation": "uresnet",
        "point_proposal": None,
        "fragmentation": None,
        "shower_aggregation": None,
        "shower_primary": None,
        "track_aggregation": None,
        "particle_aggregation": None,
        "inter_aggregation": None,
        "particle_identification": None,
        "primary_identification": None,
        "orientation_identification": None,
        "calibration": None,
    }
    loss = FullChainLoss(chain=chain)
    recorder = RecordingLoss()
    loss.uresnet_loss = recorder

    seg_label_tensor = torch.zeros((5, 2), dtype=torch.float32)
    seg_label_tensor[:, 0] = 0
    seg_label_tensor[:, SHAPE_COL] = torch.tensor([0, GHOST_SHP, 1, 2, GHOST_SHP])
    seg_label = TensorBatch(seg_label_tensor, batch_size=1, has_batch_col=True)

    segmentation = TensorBatch(
        torch.tensor(
            [
                [2.0, 0.1, 0.0],
                [0.1, 3.0, 0.0],
                [0.2, 0.3, 4.0],
            ],
            dtype=torch.float32,
        ),
        counts=torch.tensor([3]),
    )
    orig_index = IndexBatch(
        torch.tensor([0, 2, 4], dtype=torch.long),
        spans=torch.tensor([0], dtype=torch.long),
        counts=torch.tensor([3], dtype=torch.long),
    )

    loss(seg_label=seg_label, segmentation=segmentation, orig_index=orig_index)

    assert len(recorder.calls) == 1
    seg_label_used = recorder.calls[0]["seg_label"]
    segmentation_used = recorder.calls[0]["segmentation"]

    assert seg_label_used.counts.tolist() == [2]
    assert segmentation_used.counts.tolist() == [2]
    assert seg_label_used.tensor.shape[0] == 2
    assert segmentation_used.tensor.shape[0] == 2
    assert torch.equal(seg_label_used.tensor[:, SHAPE_COL], torch.tensor([0.0, 1.0]))
    assert torch.equal(segmentation_used.tensor, segmentation.tensor[:2])


def test_group_labels_accepts_shape_restriction_without_model():
    full_chain = object.__new__(FullChain)
    full_chain.fragment_shapes = [SHOWR_SHP, GHOST_SHP]

    clust_label_array = np.zeros((3, PRGRP_COL + 1), dtype=np.float32)
    clust_label_array[:, GROUP_COL] = np.array([7, 7, 9], dtype=np.float32)
    clust_label_array[:, PRGRP_COL] = np.array([1, 1, 0], dtype=np.float32)
    clust_label = TensorBatch(clust_label_array, counts=np.array([3]))

    clusts = IndexBatch(
        [np.array([0, 1], dtype=np.int64), np.array([2], dtype=np.int64)],
        spans=np.array([3]),
        counts=np.array([2]),
        single_counts=np.array([2, 1]),
    )
    clust_shapes = TensorBatch(np.array([SHOWR_SHP, GHOST_SHP]), counts=np.array([2]))

    groups, group_shapes, group_primaries, shape_index = full_chain.group_labels(
        clust_label,
        clusts,
        clust_shapes,
        shapes=[SHOWR_SHP],
        aggregate_shapes=True,
        shape_use_primary=True,
    )

    assert np.array_equal(shape_index, [0])
    assert groups.counts.tolist() == [1]
    assert np.array_equal(groups.index_list[0], [0, 1])
    assert group_shapes.tensor.tolist() == [SHOWR_SHP]
    assert group_primaries is groups


def test_label_fragmentation_reads_shapes_from_shape_column():
    full_chain = object.__new__(FullChain)
    full_chain.fragmentation = "label"
    full_chain.result = {}

    data = TensorBatch(np.zeros((4, 1), dtype=np.float32), counts=np.array([4]))
    clust_label_array = np.zeros((4, CLUST_COL + 2), dtype=np.float32)
    clust_label_array[:, CLUST_COL] = np.array([10, 10, 20, 20], dtype=np.float32)
    clust_label_array[:, SHAPE_COL] = np.array(
        [SHOWR_SHP, SHOWR_SHP, TRACK_SHP, TRACK_SHP], dtype=np.float32
    )
    clust_label = TensorBatch(clust_label_array, counts=np.array([4]))

    full_chain.run_fragmentation(data, clust_label)

    fragments = full_chain.result["fragment_clusts"]
    fragment_shapes = full_chain.result["fragment_shapes"]

    assert fragments.counts.tolist() == [2]
    assert [f.tolist() for f in fragments.index_list] == [[0, 1], [2, 3]]
    assert fragment_shapes.tensor.tolist() == [SHOWR_SHP, TRACK_SHP]


def test_build_groups_falls_back_when_primary_is_missing():
    full_chain = object.__new__(FullChain)

    clusts = IndexBatch(
        [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ],
        spans=np.array([3]),
        counts=np.array([3]),
        single_counts=np.array([1, 1, 1]),
    )
    clust_shapes = TensorBatch(
        np.array([SHOWR_SHP, SHOWR_SHP, TRACK_SHP]), counts=np.array([3])
    )
    group_pred = TensorBatch(np.array([5, 5, 9]), counts=np.array([3]))
    primary_mask = TensorBatch(np.array([False, False, True]), counts=np.array([3]))

    groups, group_shapes, group_primaries = full_chain.build_groups(
        clusts,
        clust_shapes,
        group_pred,
        primary_mask=primary_mask,
        aggregate_shapes=True,
        shape_use_primary=True,
        retain_primaries=True,
    )

    assert [g.tolist() for g in groups.index_list] == [[0, 1], [2]]
    assert group_shapes.tensor.tolist() == [SHOWR_SHP, TRACK_SHP]
    assert [p.tolist() for p in group_primaries.index_list] == [[0, 1], [2]]


def test_prepare_grappa_input_uses_label_points_without_ppn(monkeypatch):
    full_chain = object.__new__(FullChain)
    full_chain.result = {}

    model = SimpleNamespace(node_encoder=SimpleNamespace(add_points=True))
    data = TensorBatch(np.zeros((3, 4), dtype=np.float32), counts=np.array([3]))
    clusts = IndexBatch(
        [np.array([0], dtype=np.int64), np.array([1, 2], dtype=np.int64)],
        spans=np.array([3]),
        counts=np.array([2]),
        single_counts=np.array([1, 2]),
    )
    primaries = IndexBatch(
        [np.array([1], dtype=np.int64), np.array([2], dtype=np.int64)],
        spans=np.array([3]),
        counts=np.array([2]),
        single_counts=np.array([1, 1]),
    )
    clust_shapes = TensorBatch(np.array([SHOWR_SHP, SHOWR_SHP]), counts=np.array([2]))
    coord_label = TensorBatch(np.zeros((2, 9), dtype=np.float32), counts=np.array([2]))
    label_points = TensorBatch(np.ones((2, 6), dtype=np.float32), counts=np.array([2]))
    calls = []

    def fake_label_points(data_arg, coord_label_arg, clusts_arg, **kwargs):
        calls.append((data_arg, coord_label_arg, clusts_arg, kwargs))
        return label_points

    monkeypatch.setattr(
        full_chain_mod, "get_cluster_points_label_batch", fake_label_points
    )

    grappa_input = full_chain.prepare_grappa_input(
        model,
        data,
        clusts,
        clust_shapes,
        clust_primaries=primaries,
        coord_label=coord_label,
        point_use_primaries=True,
    )

    assert grappa_input["points"] is label_points
    assert "coord_label" not in grappa_input
    assert calls == [
        (data, coord_label, clusts, {"use_group": True}),
    ]
