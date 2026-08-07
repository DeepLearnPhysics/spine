"""Tests for the generic overlay helper."""

import numpy as np
import pytest

from spine.constants import SHAPE_PREC
from spine.data import (
    ClusterLabelData,
    EdgeIndexData,
    IndexData,
    IndexListData,
    Meta,
    ObjectListData,
    TensorData,
)
from spine.io.overlay import Overlayer


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
    return TensorData(
        coords=np.asarray(coords, dtype=np.int64),
        features=np.asarray(feats, dtype=np.float32),
        meta=make_meta(),
        **kwargs,
    )


def make_cluster_label(
    coord, *, particles=True, shape=0, precedence=SHAPE_PREC, meta=None
):
    """Build one compact cluster-label product for overlay tests."""
    particle_table = None
    features = [[1.0, 0.0]]
    if particles:
        features = [[1.0, 0.0, 0.0]]
        particle_table = {
            "particle": np.asarray([0], dtype=np.int64),
            "group": np.asarray([0], dtype=np.int64),
            "ancestor": np.asarray([0], dtype=np.int64),
            "interaction": np.asarray([0], dtype=np.int64),
            "nu": np.asarray([0], dtype=np.int64),
            "shape": np.asarray([shape], dtype=np.int64),
        }
    return ClusterLabelData(
        coords=np.asarray([coord], dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
        particles=particle_table,
        meta=make_meta() if meta is None else meta,
        precedence=(
            np.asarray(precedence) if particles and precedence is not None else None
        ),
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
        data_keys={"run": "scalar", "voxels": "tensor"},
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
            "edges": EdgeIndexData(
                features=np.asarray([[0], [1]], dtype=np.int64), span=2
            )
        },
        {
            "edges": EdgeIndexData(
                features=np.asarray([[0], [1]], dtype=np.int64), span=2
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"edges": "tensor"},
        methods={"edges": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    assert len(result) == 1
    assert np.array_equal(
        result[0]["edges"].features, np.asarray([[0, 2], [1, 3]], dtype=np.int64)
    )
    assert result[0]["edges"].span == 4


def test_overlayer_offsets_flat_indexes():
    """Overlay should shift flat index tensors by cumulative global shifts."""
    batch = [
        {"index": IndexData(features=np.asarray([0, 1], dtype=np.int64), span=2)},
        {"index": IndexData(features=np.asarray([0], dtype=np.int64), span=1)},
    ]
    overlay = Overlayer(
        data_keys={"index": "tensor"},
        methods={"index": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    assert len(result) == 1
    assert np.array_equal(
        result[0]["index"].features, np.asarray([0, 1, 2], dtype=np.int64)
    )
    assert result[0]["index"].span == 3


def test_overlayer_offsets_index_lists():
    """Overlay should shift each entry of a jagged index-list payload."""
    batch = [
        {
            "clusts": IndexListData(
                features=[np.asarray([0, 1]), np.asarray([1])],
                span=2,
                single_counts=np.asarray([2, 1]),
            )
        },
        {
            "clusts": IndexListData(
                features=[np.asarray([0])],
                span=2,
                single_counts=np.asarray([1]),
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"clusts": "tensor"},
        methods={"clusts": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    assert len(result) == 1
    assert isinstance(result[0]["clusts"], IndexListData)
    assert result[0]["clusts"].span == 4
    np.testing.assert_array_equal(result[0]["clusts"].features[0], np.asarray([0, 1]))
    np.testing.assert_array_equal(result[0]["clusts"].features[1], np.asarray([1]))
    np.testing.assert_array_equal(result[0]["clusts"].features[2], np.asarray([2]))
    np.testing.assert_array_equal(
        result[0]["clusts"].single_counts, np.asarray([2, 1, 1])
    )


def test_overlayer_offsets_index_lists_without_single_counts():
    """Overlay should infer index-list element sizes when single counts are absent."""
    batch = [
        {"clusts": IndexListData(features=[np.asarray([0, 1])], span=2)},
        {"clusts": IndexListData(features=[np.asarray([0])], span=2)},
    ]
    overlay = Overlayer(
        data_keys={"clusts": "tensor"},
        methods={"clusts": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)
    np.testing.assert_array_equal(result[0]["clusts"].single_counts, np.asarray([2, 1]))


def test_overlayer_rejects_mismatched_match_scalars():
    """Overlay should fail when a `match` scalar disagrees."""
    batch = [{"run": 1}, {"run": 2}]
    overlay = Overlayer(
        data_keys={"run": "scalar"},
        methods={"run": "match"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="do not match"):
        overlay(batch)


def test_overlayer_get_assignments_constant_warns():
    """Constant overlay should warn when the batch size is not divisible."""
    overlay = Overlayer(
        data_keys={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
    )

    with pytest.warns(UserWarning, match="not a divider"):
        assignments = overlay.get_assignments(3)

    assert np.array_equal(assignments, np.asarray([0, 0, 1]))


def test_overlayer_constructor_validation():
    """Overlay construction should validate mode and multiplicity eagerly."""
    with pytest.raises(ValueError, match="Overlay mode not recognized"):
        Overlayer(
            data_keys={"run": "scalar"},
            methods={"run": "cat"},
            multiplicity=1,
            mode="bad",
        )

    with pytest.raises(ValueError, match="non-zero positive integer"):
        Overlayer(
            data_keys={"run": "scalar"},
            methods={"run": "cat"},
            multiplicity=0,
        )

    with pytest.raises(ValueError, match="data_keys"):
        Overlayer(data_keys=None, methods={}, multiplicity=1)


def test_overlayer_dispatches_data_objects_and_cluster_labels():
    """Overlay should dispatch self-defining objects and compact labels."""
    meta = make_meta()
    batch = [
        {"meta": meta, "label": make_cluster_label([0, 0, 0])},
        {"meta": meta, "label": make_cluster_label([1, 1, 1])},
    ]
    overlay = Overlayer(
        data_keys=("meta", "label"),
        methods={"meta": "match", "label": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]

    assert result["meta"] == meta
    assert isinstance(result["label"], ClusterLabelData)
    np.testing.assert_array_equal(result["label"].voxel_field("cluster"), [0, 1])
    np.testing.assert_array_equal(result["label"].voxel_field("particle"), [0, 1])
    np.testing.assert_array_equal(result["label"].particles["particle"], [0, 1])
    np.testing.assert_array_equal(result["label"].particles["group"], [0, 1])
    np.testing.assert_array_equal(result["label"].particles["ancestor"], [0, 1])
    np.testing.assert_array_equal(result["label"].particles["interaction"], [0, 1])
    np.testing.assert_array_equal(result["label"].particles["nu"], [0, 1])


def test_overlayer_cluster_labels_without_particles():
    """Cluster-label overlay should support voxel-only products."""
    batch = [
        {"label": make_cluster_label([0, 0, 0], particles=False)},
        {"label": make_cluster_label([1, 1, 1], particles=False)},
    ]
    overlay = Overlayer(data_keys=("label",), methods={"label": "cat"}, multiplicity=2)

    result = overlay(batch)[0]["label"]

    assert result.particles is None
    np.testing.assert_array_equal(result.voxel_field("cluster"), [0, 1])


def test_overlayer_cluster_labels_do_not_treat_particle_indexes_as_shapes():
    """Duplicate cleanup must not apply semantic precedence to particle indexes."""
    batch = [{"label": make_cluster_label([0, 0, 0])} for _ in range(8)]
    overlay = Overlayer(
        data_keys=("label",), methods={"label": "cat"}, multiplicity=len(batch)
    )

    result = overlay(batch)[0]["label"]

    assert len(result) == 1
    np.testing.assert_array_equal(result.voxel_field("value"), [8.0])
    np.testing.assert_array_equal(result.voxel_field("cluster"), [7.0])
    np.testing.assert_array_equal(result.voxel_field("particle_index"), [7.0])


def test_overlayer_cluster_labels_apply_carried_shape_precedence():
    """The product's shape precedence should select among duplicate voxels."""
    batch = [
        {"label": make_cluster_label([0, 0, 0], shape=1, precedence=[0, 1])},
        {"label": make_cluster_label([0, 0, 0], shape=0, precedence=[0, 1])},
    ]
    overlay = Overlayer(data_keys=("label",), methods={"label": "cat"}, multiplicity=2)

    result = overlay(batch)[0]["label"]

    assert len(result) == 1
    np.testing.assert_array_equal(result.voxel_field("value"), [2.0])
    np.testing.assert_array_equal(result.voxel_field("cluster"), [1.0])
    np.testing.assert_array_equal(result.voxel_field("particle_index"), [1.0])
    np.testing.assert_array_equal(result.voxel_field("shape"), [0])


def test_overlayer_cluster_label_validation():
    """Cluster-label overlay should reject incompatible event products."""
    overlay = Overlayer(data_keys=("label",), methods={"label": "cat"}, multiplicity=2)
    label = make_cluster_label([0, 0, 0])

    with pytest.raises(TypeError, match="matching parser products"):
        overlay.stack_cluster_labels(
            [{"label": label}, {"label": object()}], "label", [0, 1]
        )

    shifted_meta = Meta(
        lower=np.asarray([1.0, 0.0, 0.0]),
        upper=np.asarray([11.0, 10.0, 10.0]),
        size=np.ones(3),
        count=np.full(3, 10),
    )
    with pytest.raises(ValueError, match="metadata must match"):
        overlay.stack_cluster_labels(
            [
                {"label": label},
                {"label": make_cluster_label([1, 1, 1], meta=shifted_meta)},
            ],
            "label",
            [0, 1],
        )

    with pytest.raises(ValueError, match="Particle information"):
        overlay.stack_cluster_labels(
            [
                {"label": label},
                {"label": make_cluster_label([1, 1, 1], particles=False)},
            ],
            "label",
            [0, 1],
        )


def test_overlayer_singleton_overlay_passthrough():
    """Single-entry overlays should be returned unchanged."""
    sample = {"run": 1}
    overlay = Overlayer(
        data_keys={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=2,
    )

    with pytest.warns(UserWarning, match="not a divider"):
        result = overlay([sample])
    assert result == [sample]


def test_overlayer_uniform_and_poisson_assignments(monkeypatch):
    """Stochastic overlay modes should assign overlay ids deterministically under mocks."""
    uniform = Overlayer(
        data_keys={"run": "scalar"},
        methods={"run": "cat"},
        multiplicity=3,
        mode="uniform",
    )
    poisson = Overlayer(
        data_keys={"run": "scalar"},
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
        data_keys={"run": "scalar"},
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
        data_keys={"value": "scalar"},
        methods={"value": "sum"},
        multiplicity=2,
    )
    overlay_first = Overlayer(
        data_keys={"value": "scalar"},
        methods={"value": "first"},
        multiplicity=2,
    )

    assert overlay_sum(batch)[0]["value"] == 3
    assert overlay_first(batch)[0]["value"] == 1


def test_overlayer_scalar_errors():
    """Scalar overlay should fail clearly on missing or invalid methods."""
    overlay_missing = Overlayer(
        data_keys={"value": "scalar"},
        methods={"value": None},
        multiplicity=2,
    )
    overlay_bad = Overlayer(
        data_keys={"value": "scalar"},
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
        data_keys={"value": "scalar"},
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
            "objs": ObjectListData([DummyIndexedObject(0)], DummyIndexedObject(0)),
        },
        {
            "obj": DummyIndexedObject(1),
            "objs": ObjectListData([DummyIndexedObject(0)], DummyIndexedObject(0)),
        },
    ]
    overlay = Overlayer(
        data_keys={"obj": "object", "objs": "object_list"},
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
        data_keys={"obj": "object"},
        methods={"obj": "match"},
        multiplicity=2,
    )
    overlay_none = Overlayer(
        data_keys={"obj": "object"},
        methods={"obj": None},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="do not match"):
        overlay_match.merge_objects(batch, "obj", [0, 1])

    with pytest.raises(ValueError, match="not specified"):
        overlay_none.merge_objects(batch, "obj", [0, 1])

    overlay_bad = Overlayer(
        data_keys={"obj": "object"},
        methods={"obj": "bad"},
        multiplicity=2,
    )
    with pytest.raises(ValueError, match="not recognized"):
        overlay_bad.merge_objects(batch, "obj", [0, 1])


def test_overlayer_merge_objects_cat_mode():
    """Object overlay should support concatenation into a ObjectListData."""
    batch = [{"obj": DummyIndexedObject(1)}, {"obj": DummyIndexedObject(2)}]
    overlay = Overlayer(
        data_keys={"obj": "object"},
        methods={"obj": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["obj"]
    assert isinstance(result, ObjectListData)
    assert [obj.index for obj in result] == [1, 2]


def test_overlayer_cat_objects_dict_shifts():
    """Object-list overlay should handle dict-based index shifts."""
    batch = [
        {
            "objs": ObjectListData(
                [DummyDictIndexedObject(0, 1)],
                DummyDictIndexedObject(0, 0),
                index_shifts={"first": 1, "second": 2},
            )
        },
        {
            "objs": ObjectListData(
                [DummyDictIndexedObject(0, 1)],
                DummyDictIndexedObject(0, 0),
                index_shifts={"first": 3, "second": 4},
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"objs": "object_list"},
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
                index_cols=np.asarray([1], dtype=np.int64),
                remove_duplicates=True,
                sum_cols=np.asarray([0], dtype=np.int64),
                prec_col=1,
                precedence=np.asarray([4, 1], dtype=np.int64),
            )
        },
        {
            "pairs": make_tensor(
                [[0, 0, 0]],
                [[7.0, 2.0]],
                index_shifts=np.asarray([3], dtype=np.int64),
                index_cols=np.asarray([1], dtype=np.int64),
                remove_duplicates=True,
                sum_cols=np.asarray([0], dtype=np.int64),
                prec_col=1,
                precedence=np.asarray([4, 1], dtype=np.int64),
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"pairs": "tensor"},
        methods={"pairs": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["pairs"]
    assert np.array_equal(result.coords, np.asarray([[0, 0, 0]]))
    assert np.array_equal(result.features, np.asarray([[12.0, 4.0]], dtype=np.float32))


def test_overlayer_applies_duplicate_selection_to_feature_only_tensors():
    """Feature-only aligned tensors should follow coordinate duplicate cleanup."""
    batch = [
        {
            "sources": make_tensor(
                [[0, 0, 0], [1, 1, 1]],
                [[0, 0], [0, 1]],
                remove_duplicates=True,
                feats_only=True,
                overlay_reference="data",
            ),
            "data": make_tensor(
                [[0, 0, 0], [1, 1, 1]],
                [[1.0], [2.0]],
                remove_duplicates=True,
                sum_cols=np.asarray([0], dtype=np.int64),
                precedence=np.asarray([0], dtype=np.int64),
            ),
        },
        {
            "sources": make_tensor(
                [[1, 1, 1], [2, 2, 2]],
                [[1, 0], [1, 1]],
                remove_duplicates=True,
                feats_only=True,
                overlay_reference="data",
            ),
            "data": make_tensor(
                [[1, 1, 1], [2, 2, 2]],
                [[3.0], [4.0]],
                remove_duplicates=True,
                sum_cols=np.asarray([0], dtype=np.int64),
                precedence=np.asarray([0], dtype=np.int64),
            ),
        },
    ]
    overlay = Overlayer(
        data_keys={"sources": "tensor", "data": "tensor"},
        methods={"sources": "cat", "data": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]
    assert np.array_equal(
        result["data"].coords,
        np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
    )
    assert np.array_equal(
        result["data"].features,
        np.asarray([[1.0], [5.0], [4.0]], dtype=np.float32),
    )
    assert np.array_equal(
        result["sources"].features,
        np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
    )


def test_overlayer_feature_only_reference_without_cleanup_stacks_features():
    """Feature-only references should be no-ops when the reference keeps all rows."""
    batch = [
        {
            "data": make_tensor([[0, 0, 0]], [[1.0]]),
            "sources": make_tensor(
                [[0, 0, 0]],
                [[0, 0]],
                remove_duplicates=True,
                feats_only=True,
                overlay_reference="data",
            ),
        },
        {
            "data": make_tensor([[1, 1, 1]], [[2.0]]),
            "sources": make_tensor(
                [[1, 1, 1]],
                [[1, 0]],
                remove_duplicates=True,
                feats_only=True,
                overlay_reference="data",
            ),
        },
    ]
    overlay = Overlayer(
        data_keys={"data": "tensor", "sources": "tensor"},
        methods={"data": "cat", "sources": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["sources"]
    assert np.array_equal(result.features, np.asarray([[0, 0], [1, 0]]))


def test_overlayer_feature_only_without_duplicate_cleanup_stacks_features():
    """Feature-only tensors should stack directly when cleanup is disabled."""
    batch = [
        {
            "sources": TensorData(
                features=np.asarray([[0, 0]], dtype=np.float32),
                feats_only=True,
            )
        },
        {
            "sources": TensorData(
                features=np.asarray([[1, 0]], dtype=np.float32),
                feats_only=True,
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"sources": "tensor"},
        methods={"sources": "cat"},
        multiplicity=2,
    )

    result = overlay(batch)[0]["sources"]
    assert np.array_equal(result.features, np.asarray([[0, 0], [1, 0]]))


def test_overlayer_feature_only_duplicate_cleanup_requires_reference():
    """Feature-only tensors should name a reference for duplicate cleanup."""
    batch = [
        {
            "sources": make_tensor(
                [[0, 0, 0]],
                [[0, 0]],
                remove_duplicates=True,
                feats_only=True,
            )
        },
        {
            "sources": make_tensor(
                [[0, 0, 0]],
                [[1, 0]],
                remove_duplicates=True,
                feats_only=True,
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"sources": "tensor"},
        methods={"sources": "cat"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="overlay_reference"):
        overlay(batch)


def test_overlayer_feature_only_reference_size_mismatch_raises():
    """Feature-only cleanup should reject stale or incompatible reference selections."""
    batch = [
        {
            "sources": TensorData(
                features=np.asarray([[0, 0]], dtype=np.float32),
                remove_duplicates=True,
                feats_only=True,
                overlay_reference="data",
            )
        }
    ]
    overlay = Overlayer(
        data_keys={"sources": "tensor"},
        methods={"sources": "cat"},
        multiplicity=1,
    )
    overlay._row_selections = {"data": (np.asarray([0], dtype=np.int64), 2)}

    with pytest.raises(ValueError, match="has 1 rows"):
        overlay.stack_feature_tensor_data(batch, "sources", [0], batch[0]["sources"])


def test_overlayer_overlay_reference_validation():
    """Overlay ordering should reject missing or cyclic overlay references."""
    missing_batch = [
        {
            "sources": TensorData(
                features=np.asarray([[0]], dtype=np.float32),
                feats_only=True,
                overlay_reference="missing",
            )
        }
    ]
    overlay = Overlayer(
        data_keys={"sources": "tensor"},
        methods={"sources": "cat"},
        multiplicity=1,
    )
    with pytest.raises(ValueError, match="not available"):
        overlay.get_overlay_order(missing_batch, [0])

    cyclic_batch = [
        {
            "a": TensorData(
                features=np.asarray([[0]], dtype=np.float32),
                feats_only=True,
                overlay_reference="b",
            ),
            "b": TensorData(
                features=np.asarray([[0]], dtype=np.float32),
                feats_only=True,
                overlay_reference="a",
            ),
        }
    ]
    overlay = Overlayer(
        data_keys={"a": "tensor", "b": "tensor"},
        methods={"a": "cat", "b": "cat"},
        multiplicity=1,
    )
    with pytest.raises(ValueError, match="Cyclic overlay reference"):
        overlay.get_overlay_order(cyclic_batch, [0])


def test_overlayer_stack_tensor_requires_index_shifts():
    """Tensor overlay should require index shifts when index columns are declared."""
    batch = [
        {
            "pairs": make_tensor(
                [[0, 0, 0]],
                [[1.0, 0.0]],
                index_cols=np.asarray([1], dtype=np.int64),
            )
        },
        {
            "pairs": make_tensor(
                [[1, 1, 1]],
                [[2.0, 0.0]],
                index_cols=np.asarray([1], dtype=np.int64),
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"pairs": "tensor"},
        methods={"pairs": "cat"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="Index shifts must be provided"):
        overlay(batch)


def test_overlayer_stack_tensor_duplicate_filter_requires_coords():
    """Duplicate filtering requires coordinates to be present."""
    batch = [
        {
            "feat": TensorData(
                features=np.asarray([[1.0]], dtype=np.float32),
                remove_duplicates=True,
            )
        },
        {
            "feat": TensorData(
                features=np.asarray([[2.0]], dtype=np.float32),
                remove_duplicates=True,
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"feat": "tensor"},
        methods={"feat": "cat"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="Must provide coordinates"):
        overlay(batch)


def test_overlayer_stack_tensors_rejects_unknown_payload():
    """Tensor overlay dispatcher should reject unknown parser payload types."""
    overlay = Overlayer(
        data_keys={"bad": "tensor"},
        methods={"bad": "cat"},
        multiplicity=2,
    )

    with pytest.raises(TypeError, match="Unsupported parser payload type"):
        overlay.stack_tensors([{"bad": object()}, {"bad": object()}], "bad", [0, 1])


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
            "voxels": TensorData(
                coords=np.asarray([[1, 1, 1]], dtype=np.int64),
                features=np.asarray([[2.0]], dtype=np.float32),
                meta=other_meta,
            )
        },
    ]
    overlay = Overlayer(
        data_keys={"voxels": "tensor"},
        methods={"voxels": "cat"},
        multiplicity=2,
    )

    with pytest.raises(ValueError, match="metadata must match"):
        overlay(batch)


def test_overlayer_direct_index_payload_stackers_cover_branches():
    """Direct stacker calls should cover index payload branch details."""
    overlay = Overlayer(
        data_keys={},
        methods={},
        multiplicity=1,
    )

    flat = overlay.stack_flat_index_data(
        [
            {"index": IndexData(np.asarray([0, -1], dtype=np.int64), span=2)},
            {"index": IndexData(np.asarray([1], dtype=np.int64), span=3)},
        ],
        "index",
        [0, 1],
        IndexData(np.asarray([0, -1], dtype=np.int64), span=2),
    )
    assert flat.span == 5
    assert np.array_equal(flat.features, np.asarray([0, -1, 3], dtype=np.int64))

    index_list = overlay.stack_index_list_data(
        [
            {"clusts": IndexListData([np.asarray([0, -1])], span=2)},
            {
                "clusts": IndexListData(
                    [np.asarray([0]), np.asarray([-1])],
                    span=3,
                    single_counts=np.asarray([1, 1]),
                )
            },
        ],
        "clusts",
        [0, 1],
        IndexListData([np.asarray([0, -1])], span=2),
    )
    assert index_list.span == 5
    assert np.array_equal(index_list.single_counts, np.asarray([2, 1, 1]))

    edge = overlay.stack_edge_index_data(
        [
            {"edge": EdgeIndexData(np.asarray([[0, -1]], dtype=np.int64), span=2)},
            {"edge": EdgeIndexData(np.asarray([[1]], dtype=np.int64), span=3)},
        ],
        "edge",
        [0, 1],
        EdgeIndexData(np.asarray([[0, -1]], dtype=np.int64), span=2),
    )
    assert edge.span == 5
    assert np.array_equal(edge.features, np.asarray([[0, -1, 3]], dtype=np.int64))
