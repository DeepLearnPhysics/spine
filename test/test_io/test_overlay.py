"""Tests for the generic overlay helper."""

import numpy as np
import pytest

from spine.data import Meta
from spine.io.overlay import Overlayer
from spine.io.parse.data import ParserTensor


def make_meta():
    """Build a simple cubic metadata object."""
    return Meta(
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
        size=np.asarray([1.0, 1.0, 1.0]),
        count=np.asarray([10, 10, 10]),
    )


def make_tensor(coords, feats, **kwargs):
    """Build a parser tensor for overlay tests."""
    return ParserTensor(
        coords=np.asarray(coords, dtype=np.int64),
        features=np.asarray(feats, dtype=np.float32),
        meta=make_meta(),
        **kwargs,
    )


def test_overlayer_merges_scalars_and_tensors():
    """Overlay should merge scalar and tensor products consistently."""
    batch = [
        {
            "run": 12,
            "voxels": make_tensor([[0, 0, 0]], [[1.0]]),
        },
        {
            "run": 12,
            "voxels": make_tensor([[1, 1, 1]], [[2.0]]),
        },
    ]
    overlay = Overlayer(
        data_types={"run": "scalar", "voxels": "tensor"},
        methods={"run": "match", "voxels": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    assert len(result) == 1
    assert result[0]["run"] == 12
    assert np.array_equal(
        result[0]["voxels"].coords, np.asarray([[0, 0, 0], [1, 1, 1]])
    )
    assert np.array_equal(
        result[0]["voxels"].features, np.asarray([[1.0], [2.0]], dtype=np.float32)
    )


def test_overlayer_offsets_index_tensors():
    """Overlay should shift feature indexes when global shifts are provided."""
    batch = [
        {
            "edges": ParserTensor(
                features=np.asarray([[0, 1]], dtype=np.int64), global_shift=2
            )
        },
        {
            "edges": ParserTensor(
                features=np.asarray([[0, 1]], dtype=np.int64), global_shift=2
            )
        },
    ]
    overlay = Overlayer(
        data_types={"edges": "tensor"},
        methods={"edges": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    assert len(result) == 1
    assert np.array_equal(
        result[0]["edges"].features, np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    )
    assert result[0]["edges"].global_shift == 4


def test_overlayer_rejects_mismatched_match_scalars():
    """Overlay should fail when a `match` scalar disagrees."""
    batch = [{"run": 1}, {"run": 2}]
    overlay = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "match"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="do not match"):
        overlay(batch)
