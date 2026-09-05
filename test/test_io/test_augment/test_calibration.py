"""Tests for calibration-parameter image augmentation."""

import numpy as np
import pytest

from spine.calib import CalibrationManager
from spine.data import RunInfo, TensorData
from spine.geo import GeoManager
from spine.io.augment import AugmentManager, CalibrationAugment

from .helpers import make_meta


@pytest.fixture(autouse=True)
def geometry():
    """Provide detector geometry for calibration managers."""
    GeoManager.reset()
    GeoManager.initialize_or_get(detector="icarus")
    yield
    GeoManager.reset()


def make_event(values=(8.0, 12.0)):
    """Build one sparse response image and aligned TPC provenance."""
    meta = make_meta()
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    data = TensorData(
        coords=coords,
        features=np.asarray(values, dtype=np.float32).reshape(-1, 1),
        meta=meta,
    )
    sources = TensorData(
        features=np.array([[0, 0], [0, 0]], dtype=np.int64), feats_only=True
    )
    return {"data": data, "sources": sources, "run_info": RunInfo(run=12)}


def test_calibration_varies_nominal_raw_gain():
    """A doubled thrown gain should halve nominal raw ADC response."""
    event = make_event()
    augment = CalibrationAugment(
        features={"data": 0},
        sources="sources",
        nominal={"gain": {"gain": 2.0}},
        throws={
            "gain": {
                "gain": {
                    "distribution": "uniform",
                    "range": [2.0, 2.0],
                    "relative": True,
                }
            }
        },
    )

    result, _ = augment(event, event["data"].meta, ["data"], {})

    np.testing.assert_allclose(result["data"].features[:, 0], [4.0, 6.0])


def test_calibration_can_select_paired_signal_response_functions():
    """Module choices should keep forward and inverse expressions paired."""
    event = make_event()
    augment = CalibrationAugment(
        features={"data": [0]},
        sources="sources",
        nominal={
            "response": {
                "response_func": "2 * x",
                "inverse_response_func": "x / 2",
            }
        },
        throws={
            "response": {
                "choices": [
                    {
                        "response_func": "4 * x",
                        "inverse_response_func": "x / 4",
                    }
                ]
            }
        },
    )

    result, _ = augment(event, event["data"].meta, ["data"], {})

    np.testing.assert_allclose(result["data"].features[:, 0], [4.0, 6.0])


def test_calibration_supports_direct_simulation_and_image_noise(monkeypatch):
    """Simulation should invert the thrown chain before shared image noise."""
    event = make_event()
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 1.0)
    augment = CalibrationAugment(
        features={"data": 0},
        sources="sources",
        mode="simulate",
        nominal={"gain": {"gain": 2.0}},
        throws={
            "gain": {
                "gain": {
                    "distribution": "uniform",
                    "range": [1.0, 1.0],
                    "relative": True,
                }
            }
        },
        noise={"scale": 1.0, "scope": "image"},
    )

    result, _ = augment(event, event["data"].meta, ["data"], {})

    np.testing.assert_allclose(result["data"].features[:, 0], [5.0, 7.0])


def test_calibration_uses_run_dependent_nominal_constant():
    """Database-like mappings should be resolved before relative throws."""
    event = make_event()
    augment = CalibrationAugment(
        features={"data": 0},
        sources="sources",
        run_info="run_info",
        nominal={"gain": {"gain_db": {12: 4.0}}},
        throws={
            "gain": {
                "gain": {
                    "distribution": "uniform",
                    "range": [0.5, 0.5],
                    "relative": True,
                }
            }
        },
    )

    result, _ = augment(event, event["data"].meta, ["data"], {})

    np.testing.assert_allclose(result["data"].features[:, 0], [16.0, 24.0])


def test_calibration_varies_lifetime_and_recombination():
    """Physical throws should compose through calibrated energy space."""
    event = make_event()
    nominal = {
        "gain": {"gain": 2.0},
        "lifetime": {"lifetime": 10.0, "driftv": 2.0},
        "recombination": {"efield": 0.5},
    }
    thrown = {
        "gain": {"gain": 2.0},
        "lifetime": {"lifetime": 5.0, "driftv": 2.0},
        "recombination": {"efield": 1.0},
    }
    augment = CalibrationAugment(
        features={"data": 0},
        sources="sources",
        nominal=nominal,
        throws={
            "lifetime": {
                "lifetime": {
                    "distribution": "uniform",
                    "range": [0.5, 0.5],
                    "relative": True,
                }
            },
            "recombination": {
                "efield": {
                    "distribution": "uniform",
                    "range": [2.0, 2.0],
                    "relative": True,
                }
            },
        },
    )

    original = event["data"].features[:, 0].copy()
    points = event["data"].coordinate_data
    sources = event["sources"].features
    _, corrected = CalibrationManager(**nominal)(
        points, original, sources, meta=event["data"].meta
    )
    _, expected = CalibrationManager(**thrown)(
        points, corrected, sources, meta=event["data"].meta, inverse=True
    )

    result, _ = augment(event, event["data"].meta, ["data"], {})

    np.testing.assert_allclose(result["data"].features[:, 0], expected, rtol=1.0e-6)
    assert not np.allclose(result["data"].features[:, 0], original)


def test_calibration_is_registered_with_augmentation_manager():
    """Dataset augmentation configuration should construct this module."""
    manager = AugmentManager(
        calibration={
            "features": {"data": 0},
            "sources": "sources",
            "nominal": {"gain": {"gain": 2.0}},
            "throws": {"gain": {"gain": {"sigma": 0.0}}},
        }
    )

    assert isinstance(manager.modules[0], CalibrationAugment)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "bad"}, "mode not recognized"),
        ({"nominal": {}}, "nominal config"),
        ({"throws": {}}, "parameter throws"),
        ({"throws": {"other": {"gain": {"sigma": 1.0}}}}, "unknown module"),
    ],
)
def test_calibration_rejects_invalid_configuration(kwargs, message):
    """Malformed calibration variation contracts should fail immediately."""
    config = {
        "features": {"data": 0},
        "nominal": {"gain": {"gain": 2.0}},
        "throws": {"gain": {"gain": {"sigma": 0.1}}},
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CalibrationAugment(**config)


def make_gain_augment(**kwargs):
    """Build a minimal gain augmenter with optional configuration changes."""
    config = {
        "features": {"data": 0},
        "nominal": {"gain": {"gain": 2.0}},
        "throws": {"gain": {"gain": {"sigma": 0.0}}},
    }
    config.update(kwargs)
    return CalibrationAugment(**config)


@pytest.mark.parametrize(
    ("features", "error"),
    [
        ({}, ValueError),
        ({1: 0}, TypeError),
        ({"data": True}, TypeError),
        ({"data": [-1]}, ValueError),
        ({"data": [0, 0]}, ValueError),
    ],
)
def test_calibration_validates_feature_selection(features, error):
    """Response selections must identify unique nonnegative columns."""
    with pytest.raises(error):
        make_gain_augment(features=features)


def test_calibration_validates_probability_and_throw_structure():
    """Application probabilities and module throw blocks are explicit."""
    with pytest.raises(ValueError, match="p.*\[0, 1\]"):
        make_gain_augment(p=2.0)
    with pytest.raises(ValueError, match="nonempty mapping"):
        make_gain_augment(throws={"gain": None})
    with pytest.raises(ValueError, match="choices.*nonempty"):
        make_gain_augment(throws={"gain": {"choices": "bad"}})
    with pytest.raises(TypeError, match="choices.*mappings"):
        make_gain_augment(throws={"gain": {"choices": [1]}})


def test_calibration_can_skip_an_event(monkeypatch):
    """The event probability should support a complete no-op."""
    event = make_event()
    augment = make_gain_augment(p=0.0)
    monkeypatch.setattr(np.random, "rand", lambda: 0.5)

    result, _ = augment(event, event["data"].meta, ["data"], {})

    assert result is event
    np.testing.assert_allclose(result["data"].features[:, 0], [8.0, 12.0])


def test_calibration_supports_direct_calibration_and_vector_features():
    """Direct correction should preserve one-dimensional feature storage."""
    event = make_event()
    event["data"].features = event["data"].features[:, 0]
    augment = make_gain_augment(mode="calibrate", sources="sources")

    result, _ = augment(event, event["data"].meta, ["data"], {})

    assert result["data"].features.ndim == 1
    np.testing.assert_allclose(result["data"].features, [16.0, 24.0])


def test_calibration_rejects_ambiguous_module_choices():
    """Whole-module alternatives cannot mix with parameter distributions."""
    event = make_event()
    augment = make_gain_augment(
        throws={"gain": {"choices": [{"gain": 3.0}], "gain": {"sigma": 1.0}}}
    )

    with pytest.raises(ValueError, match="cannot combine choices"):
        augment(event, event["data"].meta, ["data"], {})


def test_calibration_rejects_unknown_parameter():
    """Throws must target constructor parameters of their nominal module."""
    event = make_event()
    augment = make_gain_augment(throws={"gain": {"other": {"sigma": 1.0}}})

    with pytest.raises(ValueError, match="unknown calibration parameter"):
        augment(event, event["data"].meta, ["data"], {})


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        (1.0, TypeError, "must be mappings"),
        ({"choices": "bad"}, ValueError, "choices must be nonempty"),
        ({"scope": "voxel"}, ValueError, "scope"),
        ({"sigma": -1.0}, ValueError, "sigma"),
        ({"distribution": "uniform"}, ValueError, "finite two-value"),
        (
            {"distribution": "uniform", "range": [2.0, 1.0]},
            ValueError,
            "range must be ordered",
        ),
        ({"distribution": "other"}, ValueError, "not recognized"),
        ({"clip": [0.0]}, ValueError, "clip must contain two"),
        ({"clip": [2.0, 1.0]}, ValueError, "clip must be ordered"),
    ],
)
def test_calibration_parameter_sampler_rejects_invalid_specs(spec, error, message):
    """Distribution configuration should fail before producing bad constants."""
    with pytest.raises(error, match=message):
        CalibrationAugment._sample_parameter(2.0, spec, 2)


def test_calibration_parameter_sampler_covers_scopes_choices_and_clipping(
    monkeypatch,
):
    """The sampler should preserve scalar/per-TPC and relative semantics."""
    assert CalibrationAugment._sample_parameter(2.0, {"choices": [3.0]}, 2) == 3.0

    monkeypatch.setattr(
        np.random, "normal", lambda mean, sigma, size: np.array([0.5, 2.0])
    )
    relative = CalibrationAugment._sample_parameter(
        2.0,
        {"relative": True, "scope": "tpc", "sigma": 1.0, "clip": [0.8, 1.2]},
        2,
    )
    assert relative == [1.6, 2.4]

    absolute = CalibrationAugment._sample_parameter(
        [2.0, 3.0], {"scope": "tpc", "sigma": 1.0, "clip": [2.0, 4.0]}, 2
    )
    assert absolute == [2.5, 4.0]

    uniform = CalibrationAugment._sample_parameter(
        2.0, {"distribution": "uniform", "range": [3.0, 3.0]}, 2
    )
    assert uniform == 3.0


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        (None, KeyError, "missing from the event"),
        (np.ones((2, 1)), TypeError, "TensorData"),
        (TensorData(features=np.ones(2), feats_only=True), ValueError, "coordinates"),
        (
            TensorData(
                coords=np.ones((2, 3), dtype=int),
                features=np.ones((2, 1, 1)),
            ),
            ValueError,
            "one- or two-dimensional",
        ),
        (
            TensorData(
                coords=np.ones((2, 3), dtype=int),
                features=np.ones((2, 1)),
            ),
            IndexError,
            "exceed its width",
        ),
    ],
)
def test_calibration_validates_response_products(replacement, error, message):
    """Configured response products must satisfy the sparse tensor contract."""
    event = make_event()
    columns = [1] if isinstance(error, type) and error is IndexError else [0]
    augment = make_gain_augment(features={"data": columns})
    if replacement is None:
        event.pop("data")
        meta = make_meta()
    else:
        event["data"] = replacement
        meta = make_meta()

    with pytest.raises(error, match=message):
        augment(event, meta, ["data"], {})


def test_calibration_validates_source_alignment_and_shape():
    """Optional detector provenance must be a row-aligned pair tensor."""
    event = make_event()
    augment = make_gain_augment(sources="sources")
    event["sources"] = TensorData(features=np.array([[0, 0]]), feats_only=True)
    with pytest.raises(ValueError, match="sources contain 1 rows"):
        augment(event, event["data"].meta, ["data"], {})

    event = make_event()
    event.pop("sources")
    with pytest.raises(KeyError, match="source product"):
        augment(event, event["data"].meta, ["data"], {})

    event["sources"] = np.ones((2, 2))
    with pytest.raises(TypeError, match="stored in a TensorData"):
        augment(event, event["data"].meta, ["data"], {})

    event["sources"] = TensorData(features=np.ones(2), feats_only=True)
    with pytest.raises(ValueError, match="shape \(N, 2\)"):
        augment(event, event["data"].meta, ["data"], {})


def test_calibration_can_infer_sources_and_validates_run_information():
    """Sources are optional, while configured run metadata is mandatory."""
    event = make_event()
    augment = make_gain_augment()
    result, _ = augment(event, event["data"].meta, ["data"], {})
    np.testing.assert_allclose(result["data"].features[:, 0], [8.0, 12.0])

    augment = make_gain_augment(run_info="run_info")
    event.pop("run_info")
    with pytest.raises(KeyError, match="run product"):
        augment(event, event["data"].meta, ["data"], {})

    event["run_info"] = object()
    with pytest.raises(TypeError, match="provide a `run` field"):
        augment(event, event["data"].meta, ["data"], {})
