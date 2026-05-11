"""Tests for the generic overlay helper."""

import numpy as np
import pytest

from spine.data import Meta
from spine.io.overlay import Overlayer
from spine.io.parse.data import ParserObjectList, ParserTensor


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


def test_overlayer_get_assignments_constant_warns():
    """Constant overlay should warn when the batch size is not divisible."""
    overlay = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
    )

    with pytest.warns(UserWarning, match="not a divider"):
        assignments = overlay.get_assignments(3)

    assert np.array_equal(assignments, np.asarray([0, 0, 1]))


def test_overlayer_singleton_overlay_passthrough():
    """Single-entry overlays should be returned unchanged."""
    sample = {"run": 1}
    overlay = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
    )

    result = overlay([sample])
    assert result == [sample]


def test_overlayer_uniform_and_poisson_assignments(monkeypatch):
    """Stochastic overlay modes should assign overlay ids deterministically under mocks."""
    uniform = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=3,
        mode="uniform",
    )
    poisson = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
        mode="poisson",
    )

    monkeypatch.setattr(np.random, "randint", lambda low, high: 2)
    monkeypatch.setattr(np.random, "poisson", lambda lam: 2)

    assert np.array_equal(uniform.get_assignments(4), np.asarray([0, 0, 1, 1]))
    assert np.array_equal(poisson.get_assignments(4), np.asarray([0, 0, 1, 1]))


def test_overlayer_invalid_runtime_mode_raises():
    """The runtime assignment dispatcher should still reject invalid internal modes."""
    overlay = Overlayer(
        data_types={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
    )
    overlay.mode = "bad"

    with pytest.raises(ValueError, match="Overlay mode not recognized"):
        overlay.get_assignments(2)


def test_overlayer_scalar_sum_and_first():
    """Scalar overlay should support `sum` and `first` modes."""
    batch = [{"value": 1}, {"value": 2}]
    overlay_sum = Overlayer(
        data_types={"value": "scalar"},
        methods={"value": "sum"},
        multiplicity=2,
    )
    overlay_first = Overlayer(
        data_types={"value": "scalar"},
        methods={"value": "first"},
        multiplicity=2,
    )

    assert overlay_sum(batch)[0]["value"] == 3
    assert overlay_first(batch)[0]["value"] == 1


def test_overlayer_scalar_errors():
    """Scalar overlay should fail clearly on missing or invalid methods."""
    overlay_missing = Overlayer(
        data_types={"value": "scalar"},
        methods={"value": None},
        multiplicity=2,
    )
    overlay_bad = Overlayer(
        data_types={"value": "scalar"},
        methods={"value": "bad"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="not specified"):
        overlay_missing.merge_scalars([{"value": 1}, {"value": 2}], "value", [0, 1])

    with pytest.raises(ValueError, match="not recognized"):
        overlay_bad.merge_scalars([{"value": 1}, {"value": 2}], "value", [0, 1])


def test_overlayer_scalar_cat_returns_array():
    """Scalar overlay should support explicit concatenation mode."""
    overlay = Overlayer(
        data_types={"value": "scalar"},
        methods={"value": "cat"},
        multiplicity=2,
    )

    result = overlay.merge_scalars([{"value": 1}, {"value": 2}], "value", [0, 1])
    assert np.array_equal(result, np.asarray([1, 2]))


class DummyIndexedObject:
    """Object with shiftable indexes for overlay tests."""

    index_attrs = ("index",)

    def __init__(self, index):
        self.index = index

    def shift_indexes(self, shift):
        self.index += shift

    def __eq__(self, other):
        return isinstance(other, DummyIndexedObject) and self.index == other.index


class DummyDictIndexedObject:
    """Object with dict-based index shifts for overlay tests."""

    index_attrs = ("first", "second")

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def shift_indexes(self, shifts):
        self.first += shifts["first"]
        self.second += shifts["second"]


def test_overlayer_merge_objects_and_object_lists():
    """Object overlay should support matching and concatenation with shifts."""
    batch = [
        {
            "obj": DummyIndexedObject(1),
            "objs": ParserObjectList([DummyIndexedObject(0)], DummyIndexedObject(0)),
        },
        {
            "obj": DummyIndexedObject(1),
            "objs": ParserObjectList([DummyIndexedObject(0)], DummyIndexedObject(0)),
        },
    ]
    overlay = Overlayer(
        data_types={"obj": "object", "objs": "object_list"},
        methods={"obj": "match", "objs": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]
    assert result["obj"] == DummyIndexedObject(1)
    assert [obj.index for obj in result["objs"]] == [0, 1]
    assert result["objs"].index_shifts == 2


def test_overlayer_merge_objects_errors():
    """Object overlay should fail clearly on mismatched or invalid methods."""
    batch = [{"obj": DummyIndexedObject(1)}, {"obj": DummyIndexedObject(2)}]
    overlay_match = Overlayer(
        data_types={"obj": "object"},
        methods={"obj": "match"},
        multiplicity=2,
    )
    overlay_none = Overlayer(
        data_types={"obj": "object"},
        methods={"obj": None},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="do not match"):
        overlay_match.merge_objects(batch, "obj", [0, 1])

    with pytest.raises(ValueError, match="not specified"):
        overlay_none.merge_objects(batch, "obj", [0, 1])

    overlay_bad = Overlayer(
        data_types={"obj": "object"},
        methods={"obj": "bad"},
        multiplicity=2,
    )
    with pytest.raises(ValueError, match="not recognized"):
        overlay_bad.merge_objects(batch, "obj", [0, 1])


def test_overlayer_merge_objects_cat_mode():
    """Object overlay should support concatenation into a ParserObjectList."""
    batch = [{"obj": DummyIndexedObject(1)}, {"obj": DummyIndexedObject(2)}]
    overlay = Overlayer(
        data_types={"obj": "object"},
        methods={"obj": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["obj"]
    assert isinstance(result, ParserObjectList)
    assert [obj.index for obj in result] == [1, 2]


def test_overlayer_cat_objects_dict_shifts():
    """Object-list overlay should handle dict-based index shifts."""
    batch = [
        {
            "objs": ParserObjectList(
                [DummyDictIndexedObject(0, 1)],
                DummyDictIndexedObject(0, 0),
                index_shifts={"first": 1, "second": 2},
            )
        },
        {
            "objs": ParserObjectList(
                [DummyDictIndexedObject(0, 1)],
                DummyDictIndexedObject(0, 0),
                index_shifts={"first": 3, "second": 4},
            )
        },
    ]
    overlay = Overlayer(
        data_types={"objs": "object_list"},
        methods={"objs": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["objs"]
    assert [(obj.first, obj.second) for obj in result] == [(0, 1), (1, 3)]
    assert result.index_shifts == {"first": 4, "second": 6}


def test_overlayer_stack_tensor_feat_index_cols_and_duplicates():
    """Tensor overlay should shift feature index columns and remove duplicates."""
    batch = [
        {
            "pairs": make_tensor(
                [[0, 0, 0]],
                [[5.0, 1.0]],
                index_shifts=np.asarray([2], dtype=np.int64),
                index_cols=np.asarray([5], dtype=np.int64),
                remove_duplicates=True,
                sum_cols=np.asarray([4], dtype=np.int64),
                prec_col=5,
                precedence=np.asarray([4, 1], dtype=np.int64),
            )
        },
        {
            "pairs": make_tensor(
                [[0, 0, 0]],
                [[7.0, 2.0]],
                index_shifts=np.asarray([3], dtype=np.int64),
                index_cols=np.asarray([5], dtype=np.int64),
                remove_duplicates=True,
                sum_cols=np.asarray([4], dtype=np.int64),
                prec_col=5,
                precedence=np.asarray([4, 1], dtype=np.int64),
            )
        },
    ]
    overlay = Overlayer(
        data_types={"pairs": "tensor"},
        methods={"pairs": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["pairs"]
    assert np.array_equal(result.coords, np.asarray([[0, 0, 0]]))
    assert np.array_equal(result.features, np.asarray([[12.0, 4.0]], dtype=np.float32))


def test_overlayer_stack_tensor_requires_matching_meta():
    """Tensor overlay should reject mismatched metadata."""
    other_meta = Meta(
        lower=np.asarray([1.0, 0.0, 0.0]),
        upper=np.asarray([11.0, 10.0, 10.0]),
        size=np.asarray([1.0, 1.0, 1.0]),
        count=np.asarray([10, 10, 10]),
    )
    batch = [
        {"voxels": make_tensor([[0, 0, 0]], [[1.0]])},
        {
            "voxels": ParserTensor(
                coords=np.asarray([[1, 1, 1]], dtype=np.int64),
                features=np.asarray([[2.0]], dtype=np.float32),
                meta=other_meta,
            )
        },
    ]
    overlay = Overlayer(
        data_types={"voxels": "tensor"},
        methods={"voxels": "cat"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="metadata must match"):
        overlay(batch)
