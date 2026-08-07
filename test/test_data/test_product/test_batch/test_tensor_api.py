"""Extended behavioral coverage for :class:`TensorBatch`."""

import numpy as np
import pytest

from spine.data import TensorBatch, TensorData, TensorSchema


def _named_batch(meta=None):
    """Build two events with packed coordinates and named features."""
    schema = TensorSchema(
        coordinate_groups={"point": (0, 1, 2)},
        feature_fields={"score": (0,), "class_scores": (1, 2)},
    )
    data = np.asarray(
        [
            [0, 0, 1, 2, 0.1, 0.2, 0.3],
            [0, 3, 4, 5, 0.4, 0.5, 0.6],
            [1, 6, 7, 8, 0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    return TensorBatch(
        data,
        counts=[2, 1],
        has_batch_col=True,
        coord_cols=[1, 2, 3],
        schema=schema,
        meta=meta,
    )


def test_tensor_batch_named_column_and_value_access():
    """Logical accessors should map schema-relative fields to packed columns."""
    batch = _named_batch()

    assert batch.coordinate_groups == {"point": (0, 1, 2)}
    np.testing.assert_array_equal(batch.coordinate_columns(), [1, 2, 3])
    np.testing.assert_array_equal(batch.coordinate_columns("point"), [1, 2, 3])
    np.testing.assert_array_equal(batch.feature_columns(), [4, 5, 6])
    np.testing.assert_array_equal(batch.feature_columns("class_scores"), [5, 6])
    np.testing.assert_array_equal(batch.coords.data, batch.data[:, 1:4])
    np.testing.assert_array_equal(batch.coordinate_data.data, batch.data[:, 1:4])
    np.testing.assert_array_equal(batch.batch_coordinates[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(batch.features.data, batch.data[:, 4:])
    np.testing.assert_allclose(batch.feature("score").data, [[0.1], [0.4], [0.7]])
    np.testing.assert_allclose(batch.feature(0).data, [0.1, 0.4, 0.7])
    np.testing.assert_allclose(batch.feature("score").values.data, [0.1, 0.4, 0.7])

    with pytest.raises(KeyError, match="Unknown coordinate group"):
        batch.coordinate_columns("missing")
    with pytest.raises(KeyError, match="Unknown feature field"):
        batch.feature_columns("missing")
    np.testing.assert_allclose(batch.values.data, [0.1, 0.4, 0.7])


def test_tensor_batch_coordinate_and_batch_column_errors():
    """Products without coordinates or batch columns should fail explicitly."""
    feature_only = TensorBatch(np.asarray([[1, 2], [3, 4]]), counts=[2])

    assert feature_only.coordinate_data is None
    with pytest.raises(ValueError, match="no coordinate columns"):
        feature_only.coordinate_columns()
    with pytest.raises(ValueError, match="no coordinate columns"):
        _ = feature_only.batch_coordinates
    with pytest.raises(ValueError, match="no packed batch column"):
        _ = feature_only.batch_col

    ambiguous = TensorBatch(
        np.zeros((1, 7)),
        counts=[1],
        coord_cols=np.arange(6),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"value": (0,)},
        ),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous.coordinate_columns()


def test_tensor_batch_one_dimensional_feature_access():
    """Scalar feature batches should support the same named interface."""
    batch = TensorBatch(np.asarray([1.0, 2.0]), counts=[2])

    np.testing.assert_array_equal(batch.feature_columns(), [0])
    np.testing.assert_array_equal(batch.feature_columns("value"), [0])
    np.testing.assert_array_equal(batch.features.data, [1.0, 2.0])
    np.testing.assert_array_equal(batch.feature("value").data, [1.0, 2.0])
    np.testing.assert_array_equal(batch.feature(0).data, [1.0, 2.0])
    np.testing.assert_array_equal(batch.values.data, [1.0, 2.0])

    with pytest.raises(KeyError, match="Unknown feature field"):
        batch.feature_columns("missing")
    with pytest.raises(IndexError, match="only contain column 0"):
        batch.feature(1)

    malformed = TensorBatch(
        np.asarray([1.0]),
        counts=[1],
        schema=TensorSchema(feature_fields={"bad": (1,)}, feats_only=True),
    )
    with pytest.raises(IndexError, match="only contain column 0"):
        malformed.feature("bad")

    empty = TensorBatch(np.empty((2, 0)), counts=[2])
    with pytest.raises(ValueError, match="at least one feature column"):
        _ = empty.values


def test_tensor_batch_event_mask_selection_and_split():
    """Row operations should update counts while preserving logical metadata."""
    batch = _named_batch(meta=["first", "second"])
    event = batch.event(1)
    assert event.meta == "second"
    np.testing.assert_array_equal(event.coords, [[6, 7, 8]])
    np.testing.assert_allclose(event.features, [[0.7, 0.8, 0.9]])

    selected = batch.select(np.asarray([False, True, True]))
    np.testing.assert_array_equal(selected.counts, [1, 1])
    assert selected.schema == batch.schema
    assert selected.meta == batch.meta

    batch.apply_mask(np.asarray([True, False, True]))
    np.testing.assert_array_equal(batch.counts, [1, 1])
    np.testing.assert_array_equal(batch.edges, [0, 1, 2])
    split = batch.split()
    assert len(split) == 2
    np.testing.assert_array_equal(split[0], batch.data[:1])


def test_tensor_batch_event_rejects_one_dimensional_coordinates():
    """A one-dimensional packed product cannot also advertise coordinates."""
    batch = TensorBatch(
        np.asarray([1, 2]),
        counts=[2],
        coord_cols=[0],
        schema=TensorSchema(coordinate_groups={"coordinate": (0,)}),
    )
    with pytest.raises(ValueError, match="One-dimensional"):
        batch.event(0)


def test_tensor_batch_merge_and_schema_validation():
    """Entry-wise merges should preserve compatible schemas and metadata."""
    left = TensorBatch(np.asarray([[1], [2], [3]]), counts=[2, 1], meta=["a", "b"])
    right = TensorBatch(np.asarray([[4], [5], [6]]), counts=[1, 2], meta=["a", "b"])
    merged = left.merge(right)

    np.testing.assert_array_equal(merged.data[:, 0], [1, 2, 4, 3, 5, 6])
    np.testing.assert_array_equal(merged.counts, [3, 3])
    assert merged.meta == left.meta

    incompatible = TensorBatch(
        np.asarray([[4], [5], [6]]),
        counts=[1, 2],
        schema=TensorSchema(feature_fields={"other": (0,)}, feats_only=True),
    )
    with pytest.raises(ValueError, match="different schemas"):
        left.merge(incompatible)


class _Meta:
    """Minimal reversible coordinate metadata for conversion tests."""

    @staticmethod
    def to_cm(values, center=True):
        assert center is True
        return values * 2

    @staticmethod
    def to_px(values, floor=True):
        assert floor is True
        return values / 2


def test_tensor_batch_coordinate_unit_conversion():
    """Every named coordinate group should be converted independently."""
    schema = TensorSchema(
        coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
        feature_fields={"value": (0,)},
    )
    data = np.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
    batch = TensorBatch(data, counts=[1], coord_cols=np.arange(6), schema=schema)

    batch.to_cm(_Meta())
    np.testing.assert_array_equal(batch.data[0, :6], [2, 4, 6, 8, 10, 12])
    batch.to_px(_Meta())
    np.testing.assert_array_equal(batch.data[0, :6], [1, 2, 3, 4, 5, 6])

    feature_only = TensorBatch(np.ones((1, 1)), counts=[1])
    with pytest.raises(ValueError, match="without coordinate metadata"):
        feature_only.to_cm(_Meta())
    with pytest.raises(ValueError, match="without coordinate metadata"):
        feature_only.to_px(_Meta())


def test_tensor_batch_factories_cover_feature_and_coordinate_products():
    """Event-list factories should preserve schemas, counts and metadata."""
    with pytest.raises(ValueError, match="at least one tensor"):
        TensorBatch.from_list([])
    raw = TensorBatch.from_list([np.asarray([[1], [2]]), np.asarray([[3]])])
    np.testing.assert_array_equal(raw.counts, [2, 1])

    feature_events = [
        TensorData(np.asarray([1.0, 2.0]), meta="a"),
        TensorData(np.asarray([3.0]), meta="b"),
    ]
    features = TensorBatch.from_data_list(feature_events)
    np.testing.assert_array_equal(features.data, [1, 2, 3])
    assert features.meta == ["a", "b"]

    coordinate_events = [
        TensorData(
            np.asarray([1.0, 2.0]),
            np.asarray([[0, 1, 2], [3, 4, 5]]),
            meta="a",
        ),
        TensorData(np.asarray([3.0]), np.asarray([[6, 7, 8]]), meta="b"),
    ]
    coordinates = TensorBatch.from_data_list(coordinate_events)
    assert coordinates.has_batch_col is True
    np.testing.assert_array_equal(coordinates.counts, [2, 1])
    np.testing.assert_array_equal(coordinates.data[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(coordinates.event(1).coords, [[6, 7, 8]])

    scalar_coordinates = TensorBatch.from_data_list(
        [
            TensorData(np.asarray([1.0]), np.asarray([2.0])),
            TensorData(np.asarray([3.0]), np.asarray([4.0])),
        ]
    )
    np.testing.assert_array_equal(scalar_coordinates.data, [[0, 2, 1], [1, 4, 3]])

    with pytest.raises(ValueError, match="at least one event"):
        TensorBatch.from_data_list([])
    with pytest.raises(ValueError, match="different schemas"):
        TensorBatch.from_data_list(
            [TensorData(np.ones((1, 1))), TensorData(np.ones((1, 2)))]
        )
    with pytest.raises(ValueError, match="Coordinate presence"):
        schema = TensorSchema(feature_fields={"value": (0,)}, feats_only=True)
        TensorBatch.from_data_list(
            [
                TensorData(np.ones((1, 1)), schema=schema),
                TensorData(np.ones((1, 1)), np.zeros((1, 3)), schema=schema),
            ]
        )


def test_torch_tensor_batch_paths_and_mixed_backend_rejection():
    """PyTorch batches should mirror NumPy behavior without implicit mixing."""
    torch = pytest.importorskip("torch")
    numpy_batch = _named_batch()
    tensor_batch = numpy_batch.to_tensor(dtype=torch.float32, device="cpu")

    assert tensor_batch.to_tensor() is tensor_batch
    assert tensor_batch.torch_tensor() is tensor_batch.data
    np.testing.assert_array_equal(
        tensor_batch.batch_coordinates, numpy_batch.batch_coordinates
    )
    restored = tensor_batch.to_numpy()
    assert restored.to_numpy() is restored
    np.testing.assert_array_equal(restored.data, numpy_batch.data)

    with pytest.raises(TypeError, match="not backed by a numpy.ndarray"):
        tensor_batch.numpy_tensor()
    with pytest.raises(ValueError, match="numpy arrays"):
        tensor_batch.to_cm(_Meta())
    with pytest.raises(ValueError, match="numpy arrays"):
        tensor_batch.to_px(_Meta())

    torch_events = [
        TensorData(torch.tensor([1.0]), torch.tensor([[0.0, 1.0, 2.0]])),
        TensorData(torch.tensor([2.0]), torch.tensor([[3.0, 4.0, 5.0]])),
    ]
    built = TensorBatch.from_data_list(torch_events)
    assert torch.equal(built.data[:, 0], torch.tensor([0.0, 1.0]))
    raw = TensorBatch.from_list([torch.tensor([[1.0]]), torch.tensor([[2.0]])])
    assert torch.equal(raw.data, torch.tensor([[1.0], [2.0]]))

    with pytest.raises(ValueError, match="share an array backend"):
        TensorBatch.from_data_list(
            [TensorData(np.asarray([1.0])), TensorData(torch.tensor([2.0]))]
        )
