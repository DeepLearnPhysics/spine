"""Tests for detector-response augmentation."""

import numpy as np
import pytest

from spine.data import TensorData
from spine.io.augment import AugmentManager, ResponseAugment

from .helpers import make_meta


def make_response_tensor(features, coords=None):
    """Build a response tensor with optional sparse coordinates."""
    features = np.asarray(features)
    if coords is None:
        return TensorData(features=features, feats_only=True)
    return TensorData(
        coords=np.asarray(coords, dtype=np.int64),
        features=features,
        meta=make_meta(),
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"features": {}}, ValueError),
        ({"features": []}, ValueError),
        ({"features": {1: [0]}}, TypeError),
        ({"features": {"data": []}}, ValueError),
        ({"features": {"data": [[0]]}}, ValueError),
        ({"features": {"data": [[0], [0, 1]]}}, TypeError),
        ({"features": {"data": [-1]}}, ValueError),
        ({"features": {"data": [0, 0]}}, ValueError),
        ({"features": {"data": [0.5]}}, TypeError),
        ({"features": {"data": ["charge"]}}, TypeError),
        ({"features": {"data": [0]}, "gain_range": [1.0]}, ValueError),
        ({"features": {"data": [0]}, "gain_range": [np.nan, 1.0]}, ValueError),
        ({"features": {"data": [0]}, "gain_range": [-1.0, 1.0]}, ValueError),
        ({"features": {"data": [0]}, "gain_range": [2.0, 1.0]}, ValueError),
        ({"features": {"data": [0]}, "noise_sigma": -1.0}, ValueError),
        ({"features": {"data": [0]}, "dropout_prob": 1.1}, ValueError),
        ({"features": {"data": [0]}, "p": -0.1}, ValueError),
        (
            {
                "features": {"data": [0]},
                "clip_min": 2.0,
                "saturation": 1.0,
            },
            ValueError,
        ),
        ({"features": {"data": [0]}, "threshold": np.inf}, ValueError),
        ({"features": {"data": [0]}, "fill_value": np.nan}, ValueError),
    ],
)
def test_response_rejects_invalid_configuration(kwargs, error):
    """Response controls should reject ambiguous or nonphysical values."""
    with pytest.raises(error):
        ResponseAugment(**kwargs)


def test_response_applies_gain_threshold_and_saturation():
    """Deterministic response operations should preserve rows and coordinates."""
    coords = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    tensor = make_response_tensor([[1.0, 10.0], [2.0, 20.0]], coords)
    original_coords = tensor.coords.copy()
    augment = ResponseAugment(
        features={"data": 0},
        gain_range=(2.0, 2.0),
        threshold=2.5,
        saturation=3.0,
        fill_value=-1.0,
    )

    result, meta = augment({"data": tensor}, tensor.meta, ["data"], {})

    np.testing.assert_array_equal(result["data"].coords, original_coords)
    np.testing.assert_allclose(result["data"].features, [[-1.0, 10.0], [3.0, 20.0]])
    assert meta is tensor.meta


def test_response_applies_noise_and_lower_clipping(monkeypatch):
    """Additive noise should be applied before configured lower clipping."""
    tensor = make_response_tensor([[1.0, 2.0], [3.0, 4.0]])
    augment = ResponseAugment(
        features={"features": [0, 1]},
        noise_sigma=2.0,
        clip_min=0.0,
    )
    monkeypatch.setattr(
        np.random,
        "normal",
        lambda mean, sigma, shape: np.asarray([[-2.0, 1.0], [1.0, -5.0]]),
    )

    result, _ = augment({"features": tensor}, make_meta(), [], {})

    np.testing.assert_allclose(result["features"].features, [[0.0, 3.0], [4.0, 0.0]])


def test_response_shares_dropout_by_coordinate(monkeypatch):
    """Matching points should receive one signal-loss decision across products."""
    first = make_response_tensor(
        [[1.0], [2.0]],
        [[0, 0, 0], [1, 0, 0]],
    )
    second = make_response_tensor(
        [[20.0], [10.0]],
        [[1, 0, 0], [0, 0, 0]],
    )
    samples = iter((0.0, 0.1, 0.9))
    monkeypatch.setattr(np.random, "rand", lambda: next(samples))
    augment = ResponseAugment(
        features={"first": [0], "second": [0]},
        dropout_prob=0.5,
    )

    result, _ = augment(
        {"first": first, "second": second},
        first.meta,
        ["first", "second"],
        {},
    )

    np.testing.assert_allclose(result["first"].features[:, 0], [0.0, 2.0])
    np.testing.assert_allclose(result["second"].features[:, 0], [20.0, 0.0])


def test_response_handles_one_dimensional_coordinate_free_features(monkeypatch):
    """Feature vectors should retain their original dimensionality."""
    tensor = make_response_tensor([1.0, 2.0, 3.0])
    augment = ResponseAugment(features={"data": [0]}, dropout_prob=0.5)
    samples = iter((0.0, np.asarray([0.1, 0.8, 0.2])))
    monkeypatch.setattr(np.random, "rand", lambda *shape: next(samples))

    result, _ = augment({"data": tensor}, make_meta(), [], {})

    assert result["data"].features.ndim == 1
    np.testing.assert_allclose(result["data"].features, [0.0, 2.0, 0.0])


def test_response_event_probability_can_skip_augmentation(monkeypatch):
    """The event-level probability should permit a no-op response path."""
    tensor = make_response_tensor([[1.0]])
    augment = ResponseAugment(features={"data": [0]}, gain_range=(2.0, 2.0), p=0.5)
    monkeypatch.setattr(np.random, "rand", lambda: 0.75)

    data = {"data": tensor}
    result, _ = augment(data, make_meta(), [], {})

    assert result is data
    np.testing.assert_allclose(tensor.features, [[1.0]])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "missing"),
        ({"data": np.ones((2, 1))}, "TensorData"),
        ({"data": make_response_tensor(np.ones((1, 1, 1)))}, "one- or two"),
        ({"data": make_response_tensor(np.ones((2, 1)))}, "feature width"),
    ],
)
def test_response_validates_event_products(data, message):
    """Configured event products must satisfy the response tensor contract."""
    columns = [1] if "data" in data and isinstance(data["data"], TensorData) else [0]
    augment = ResponseAugment(features={"data": columns})

    with pytest.raises((KeyError, TypeError, ValueError, IndexError), match=message):
        augment(data, make_meta(), [], {})


def test_response_is_available_through_augmentation_manager():
    """Dataset augmentation configuration should construct response modules."""
    meta = make_meta()
    tensor = TensorData(
        coords=np.asarray([[0, 0, 0]], dtype=np.int64),
        features=np.asarray([[2.0]], dtype=np.float32),
        meta=meta,
    )
    manager = AugmentManager(
        response={"features": {"data": [0]}, "gain_range": [1.5, 1.5]}
    )

    result = manager({"data": tensor, "meta": meta})

    assert isinstance(manager.modules[0], ResponseAugment)
    np.testing.assert_allclose(result["data"].features, [[3.0]])
