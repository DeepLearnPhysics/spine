"""Tests for electromagnetic-shower parametrizations and energy fitting."""

from types import SimpleNamespace

import numpy as np
import pytest

import spine.physics.shower as shower


@pytest.fixture
def boundaries():
    """Return two adjacent, axis-aligned detector boxes."""
    return np.array(
        [
            [[-10.0, 10.0], [-10.0, 10.0], [0.0, 30.0]],
            [[-10.0, 10.0], [-10.0, 10.0], [30.0, 60.0]],
        ]
    )


def test_shower_energy_fitter_validation(boundaries):
    """The fitter should reject malformed or unphysical configuration."""
    invalid = [
        ({"boundaries": np.zeros((2, 2))}, "shape"),
        ({"boundaries": boundaries, "n_points": 0}, "n_points"),
        ({"boundaries": boundaries, "sigma_floor": -1.0}, "sigma_floor"),
        ({"boundaries": boundaries[:, :, ::-1]}, "widths"),
        ({"boundaries": boundaries, "energy_bounds": (2.0, 1.0)}, "bounds"),
        (
            {
                "boundaries": boundaries,
                "energy_bounds": (100.0, 2000.0),
                "use_gp": True,
            },
            "use_gp",
        ),
    ]
    for kwargs, match in invalid:
        with pytest.raises(ValueError, match=match):
            shower.ShowerEnergyFitter(**kwargs)

    for start, direction, match in [
        (np.zeros(2), np.ones(3), "shower_start"),
        (np.zeros(3), np.ones(2), "direction"),
        (np.zeros(3), np.zeros(3), "non-zero"),
    ]:
        with pytest.raises(ValueError, match=match):
            shower.ShowerEnergyFitter._validate_shower_inputs(start, direction)


def test_shower_energy_fitter_predictions_and_fit(boundaries, monkeypatch):
    """Box counting, uncertainty propagation, chi-square, and fit should work."""
    fitter = shower.ShowerEnergyFitter(
        boundaries, n_points=64, sigma_floor=2.0, energy_bounds=(10.0, 100.0)
    )
    assert fitter.n_boxes == 2

    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 30.0], [20.0, 0.0, 1.0]])
    np.testing.assert_array_equal(fitter.count_points_in_boxes(points), [1, 1])

    pred, sigma, counts = fitter.predict_box_energy(
        50.0, np.zeros(3), np.array([0.0, 0.0, 1.0])
    )
    assert pred.shape == sigma.shape == counts.shape == (2,)
    assert np.all(sigma >= 2.0)
    assert np.isfinite(fitter.chi2(50.0, pred, np.zeros(3), [0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="reco_box_energy"):
        fitter.chi2(50.0, np.ones(3), np.zeros(3), [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="reco_box_energy"):
        fitter.fit(np.ones(3), np.zeros(3), [0.0, 0.0, 1.0])

    calls = {}

    def fake_minimize(function, bounds, method, args, options):
        calls.update(bounds=bounds, method=method, options=options)
        assert np.isfinite(function(25.0, *args))
        return SimpleNamespace(x=25.0)

    monkeypatch.setattr(shower, "minimize_scalar", fake_minimize)
    assert fitter.fit(pred, np.zeros(3), [0.0, 0.0, 1.0]) == 25.0
    assert calls == {
        "bounds": (10.0, 100.0),
        "method": "bounded",
        "options": {"xatol": 1.0},
    }

    fitter.xatol = None
    fitter.fit(pred, np.zeros(3), [0.0, 0.0, 1.0])
    assert calls["options"] == {}


def test_sampling_and_shower_profiles():
    """Both longitudinal models and transverse basis choices should be finite."""
    rng = np.random.default_rng(7)
    for direction, use_gp, energy in [
        (np.array([1.0, 0.0, 0.0]), False, 500.0),
        (np.array([0.0, 0.0, 1.0]), True, 2000.0),
    ]:
        points = shower.sample_shower_points(
            32, energy, np.ones(3), direction, rng, use_gp
        )
        assert points.shape == (32, 3)
        assert np.isfinite(points).all()

    with pytest.raises(ValueError, match="non-zero"):
        shower.sample_shower_points(1, 500.0, np.zeros(3), np.zeros(3), rng)

    depths = np.array([-1.0, 5.0, 20.0])
    radii = np.array([0.1, 1.0, 2.0])
    for use_gp, energy in [(False, 500.0), (True, 2000.0)]:
        assert np.isfinite(shower.shower_long_profile(depths, energy, use_gp)).all()
        assert np.isfinite(
            shower.shower_energy_density(depths, radii, energy, use_gp)
        ).all()
        assert shower.shower_long_quantile(energy, 0.5, use_gp) > 0.0

    density = shower.shower_energy_density_3d(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        np.zeros(3),
        np.array([0.0, 0.0, 2.0]),
        500.0,
    )
    assert density.shape == (2,)
    with pytest.raises(ValueError, match="non-zero"):
        shower.shower_energy_density_3d(
            np.zeros((1, 3)), np.zeros(3), np.zeros(3), 500.0
        )


def test_shower_parameter_helpers():
    """All scalar and vector profile parameter helpers should be well behaved."""
    assert shower.shower_long_mode_gp(2000.0) > 0.0
    assert shower.shower_long_maximum_lar(500.0) > 0.0
    for params in (
        shower.shower_long_params_gp(2000.0),
        shower.shower_long_params_lar(500.0),
    ):
        assert len(params) == 2
        assert np.all(np.asarray(params) > 0.0)

    core, tail, weight = shower.shower_transverse_params_gp(
        np.array([0.0, 10.0]), 2000.0
    )
    assert np.all(core > 0.0)
    assert np.all(tail > 0.0)
    assert np.all((weight >= 0.0) & (weight <= 1.0))
    assert np.isfinite(
        shower.shower_trans_profile(np.array([0.1, 1.0]), 5.0, 500.0)
    ).all()

    energy = 500.0
    params = shower.shower_angle_params(energy)
    assert len(params) == 3
    assert np.isfinite(shower.shower_angle_profile(np.array([0.1, 0.2]), energy)).all()
    assert shower.shower_angle_mode(energy) > 0.0
    assert shower.shower_angle_quantile(energy, 0.5) > 0.0
