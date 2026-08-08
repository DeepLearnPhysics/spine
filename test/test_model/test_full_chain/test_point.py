"""Tests for aligned full-chain voxel representations."""

import numpy as np
import pytest
import torch

from spine.data import TensorBatch
from spine.model.full_chain import PointBatch


def make_data(values, counts=(2, 2)):
    """Build a two-event sparse tensor with one logical feature."""
    rows = torch.zeros((len(values), 5), dtype=torch.float32)
    rows[:, 0] = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts)
    )
    rows[:, 1] = torch.arange(len(values))
    rows[:, 4] = torch.as_tensor(values)
    return TensorBatch(
        rows,
        counts=counts,
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


def test_calibrated_points_remain_aligned_through_selection():
    """Later row selections should affect charge, energy and sources together."""
    charge = make_data([1, 2, 3, 4])
    sources = TensorBatch(torch.arange(8).reshape(4, 2), counts=[2, 2])
    calibrated = make_data([10, 20, 30, 40])
    calibrated.data[:, 1] += 100

    point_data = PointBatch.from_input(charge, sources=sources)
    point_data = point_data.with_calibration(calibrated)
    point_data = point_data.select(torch.tensor([True, False, False, True]))

    assert point_data.data is point_data.data_calib
    assert point_data.data_q.feature(0).data.tolist() == [1, 4]
    assert point_data.data_calib.feature(0).data.tolist() == [10, 40]
    assert point_data.data_calib.coordinates().data[:, 0].tolist() == [100, 103]
    assert point_data.sources.data.tolist() == [[0, 1], [6, 7]]
    assert point_data.orig_index.index.tolist() == [0, 3]
    assert point_data.data.counts.tolist() == [1, 1]

    outputs = point_data.public_outputs()
    assert outputs["data_adapt"] is point_data.data_q
    assert outputs["data_calib"] is point_data.data_calib
    assert outputs["sources_adapt"] is point_data.sources
    products = point_data.canonical_products()
    assert products["sources"] is point_data.sources


def test_point_selections_compose_original_indexes():
    """Repeated adaptations should retain a mapping to the driver input rows."""
    point_data = PointBatch.from_input(make_data([1, 2, 3, 4]))
    point_data = point_data.select(torch.tensor([True, False, True, True]))
    point_data = point_data.select(torch.tensor([False, True, True]))

    assert point_data.orig_index.index.tolist() == [2, 3]
    assert point_data.data_q.feature(0).data.tolist() == [3, 4]
    assert point_data.data.counts.tolist() == [0, 2]


def test_point_charge_replacement_copies_input():
    """Charge rescaling should preserve the driver-owned tensor."""
    original = make_data([1, 2, 3, 4])
    initial = PointBatch.from_input(original)
    assert initial.public_outputs() == {}
    point_data = initial.with_charge([5, 6, 7, 8])

    assert original.feature(0).data.tolist() == [1, 2, 3, 4]
    assert point_data.data is point_data.data_q
    assert point_data.data_q.feature(0).data.tolist() == [5, 6, 7, 8]


def test_point_alignment_uses_composed_original_indexes():
    """Original-domain labels should map onto the current selected rows."""
    original = make_data([10, 20, 30, 40])
    point_data = PointBatch.from_input(make_data([1, 2, 3, 4]))

    assert point_data.align(original) is original
    with pytest.raises(ValueError, match="without `orig_index`"):
        point_data.align(make_data([10, 20], counts=(1, 1)))

    point_data = point_data.select(np.array([True, False, True, False]))
    aligned = point_data.align(original)

    assert aligned.feature(0).data.tolist() == [10, 30]
    assert aligned.counts.tolist() == [1, 1]
    products = point_data.canonical_products()
    assert products["data"] is point_data.data
    assert products["orig_index"] is point_data.orig_index


@pytest.mark.parametrize(
    ("point_factory", "message"),
    [
        (
            lambda data: PointBatch(
                data=data,
                data_q=make_data([1, 2], counts=(1, 1)),
            ),
            "different row count",
        ),
        (
            lambda data: PointBatch(
                data=data,
                data_q=make_data([1, 2, 3, 4], counts=(1, 3)),
            ),
            "different event counts",
        ),
        (
            lambda data: PointBatch(data=data, data_q=make_data([1, 2, 3, 4])),
            "Uncalibrated active",
        ),
        (
            lambda data: PointBatch(
                data=data,
                data_q=data,
                data_calib=make_data([10, 20, 30, 40]),
            ),
            "Calibrated active",
        ),
    ],
)
def test_point_validation_rejects_misaligned_products(point_factory, message):
    """Invalid bundles should report the violated alignment invariant."""
    with pytest.raises(ValueError, match=message):
        point_factory(make_data([1, 2, 3, 4]))._validate()


def test_charge_cannot_change_after_calibration():
    """Changing charge behind an existing calibration should be rejected."""
    point_data = PointBatch.from_input(make_data([1, 2, 3, 4]))
    point_data = point_data.with_calibration(make_data([10, 20, 30, 40]))

    with pytest.raises(ValueError, match="without recalibrating"):
        point_data.with_charge(torch.ones(4))
