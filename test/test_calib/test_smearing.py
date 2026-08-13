import numpy as np
import pytest

from spine.calib.smearing import SmearingCalibrator


def test_smearing_calibrator_applies_multiplicative_normal_smearing(monkeypatch):
    samples = np.array([0.8, 1.3])
    monkeypatch.setattr(np.random, "normal", lambda **kwargs: samples)
    calibrator = SmearingCalibrator(scale=0.1, mode="multiplicative", mean=1.0)
    values = np.array([10.0, 20.0])

    result = calibrator.process(values)

    assert np.allclose(result, values * samples)


def test_smearing_calibrator_applies_additive_smearing(monkeypatch):
    samples = np.array([-2.0, 3.0])
    monkeypatch.setattr(np.random, "normal", lambda **kwargs: samples)
    calibrator = SmearingCalibrator(scale=1.0)
    values = np.array([10.0, 20.0])

    result = calibrator.process(values)

    assert np.allclose(result, values + samples)


def test_smearing_calibrator_configures_distribution(monkeypatch):
    arguments = {}

    def sample_normal(**kwargs):
        arguments.update(kwargs)
        return np.zeros(kwargs["size"])

    monkeypatch.setattr(np.random, "normal", sample_normal)
    calibrator = SmearingCalibrator(scale=0.2, mean=0.1)
    values = np.ones(3)

    calibrator.process(values)

    assert arguments == {"loc": 0.1, "scale": 0.2, "size": values.shape}


def test_smearing_calibrator_clips_and_preserves_dtype(monkeypatch):
    monkeypatch.setattr(np.random, "normal", lambda **kwargs: np.array([-2.0, 0.5]))
    calibrator = SmearingCalibrator(
        scale=1.0, mode="multiplicative", mean=1.0, clip_min=0.0
    )
    values = np.array([10.0, 20.0], dtype=np.float32)

    result = calibrator.process(values)

    assert result.dtype == values.dtype
    assert np.allclose(result, [0.0, 10.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distribution": "uniform"}, "distribution not recognized"),
        ({"mode": "relative"}, "mode not recognized"),
        ({"scale": -0.1}, "scale must be"),
        ({"scale": np.inf}, "scale must be"),
        ({"mean": np.nan}, "mean must be"),
        ({"clip_min": np.inf}, "lower bound must be"),
    ],
)
def test_smearing_calibrator_validates_configuration(kwargs, message):
    config = {"scale": 0.1, **kwargs}

    with pytest.raises(ValueError, match=message):
        SmearingCalibrator(**config)
