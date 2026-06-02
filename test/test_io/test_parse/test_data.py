"""Tests for generic parser output data structures."""

import numpy as np

from spine.constants import VALUE_COL
from spine.data import Particle
from spine.io.parse.data import (
    ParserEdgeIndex,
    ParserIndex,
    ParserIndexList,
    ParserObjectList,
    ParserTensor,
)


class DummyObject:
    """Minimal object type for ParserObjectList tests."""

    index_attrs = ()


def test_parser_tensor_feature_index_helpers():
    """ParserTensor should translate absolute columns to feature columns."""
    tensor = ParserTensor(
        features=np.ones((2, 4), dtype=np.float32),
        index_cols=np.array([VALUE_COL + 1, VALUE_COL + 3], dtype=np.int64),
        sum_cols=np.array([VALUE_COL + 2], dtype=np.int64),
        avg_cols=np.array([VALUE_COL + 0], dtype=np.int64),
        prec_col=VALUE_COL + 4,
    )

    assert np.array_equal(tensor.feat_index_cols, np.array([1, 3], dtype=np.int64))
    assert np.array_equal(tensor.feat_sum_cols, np.array([2], dtype=np.int64))
    assert np.array_equal(tensor.feat_avg_cols, np.array([0], dtype=np.int64))
    assert tensor.feat_prec_col == 4


def test_parser_tensor_feature_helpers_preserve_none_and_negative():
    """ParserTensor helper accessors should preserve sentinel values."""
    tensor = ParserTensor(
        features=np.ones((1, 2), dtype=np.float32),
        index_cols=None,
        sum_cols=None,
        prec_col=-1,
    )

    assert tensor.feat_index_cols is None
    assert tensor.feat_sum_cols is None
    assert tensor.feat_prec_col == -1


def test_parser_object_list_defaults_and_cast():
    """ParserObjectList should retain index shifts and cast back cleanly."""
    objects = ParserObjectList([DummyObject(), DummyObject()], DummyObject())
    assert objects.index_shifts == 2

    shifted = ParserObjectList([DummyObject()], DummyObject(), index_shifts={"a": 3})
    assert shifted.index_shifts == {"a": 3}
    casted = shifted.to_object_list
    assert list(casted) == list(shifted)
    assert casted.default == shifted.default


def test_parser_index_payloads_store_specialized_contracts():
    """Index-style parser payloads should preserve their specialized shape."""
    index = ParserIndex(features=np.asarray([0, 2, 4]), span=5)
    index_list = ParserIndexList(
        features=[np.asarray([0, 2]), np.asarray([1])],
        span=3,
        single_counts=np.asarray([2, 1]),
    )
    edge_index = ParserEdgeIndex(
        features=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        span=3,
    )

    np.testing.assert_array_equal(index.features, np.asarray([0, 2, 4]))
    assert index.span == 5
    assert len(index_list.features) == 2
    np.testing.assert_array_equal(index_list.single_counts, np.asarray([2, 1]))
    np.testing.assert_array_equal(
        edge_index.features, np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    )


def test_data_objects_expose_public_index_attrs():
    """Data objects should expose index metadata through a public property."""
    particle = Particle()

    assert "id" in particle.index_attrs
    assert "parent_id" in particle.index_attrs
    assert "children_id" in particle.index_attrs
