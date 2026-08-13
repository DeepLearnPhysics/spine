from types import SimpleNamespace

import numpy as np
import pytest

import spine.physics.mcs as mcs_mod
from spine.physics.mcs import (
    angles_atan2,
    angles_cos,
    angles_kahan,
    highland,
    mcs_angles,
    mcs_angles_proj,
    mcs_fit,
    mcs_nll_lar,
    split_angles,
)


def test_mcs_fit_rejects_resolution_floor_solution():
    """Angles below resolution must not become the arbitrary upper bound."""
    theta = np.zeros(4, dtype=np.float64)

    assert np.isnan(mcs_fit(theta, 105.658, 5.0))
    assert mcs_fit(theta, 105.658, 5.0, return_invalid=True) > 99999.0


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(x=10.0, success=True),
        SimpleNamespace(x=100000.0, success=True),
        SimpleNamespace(x=100.0, success=False),
        SimpleNamespace(x=np.nan, success=True),
    ],
)
def test_mcs_fit_rejects_invalid_optimizer_results(monkeypatch, result):
    monkeypatch.setattr(
        mcs_mod.scipy.optimize, "minimize_scalar", lambda *args, **kwargs: result
    )

    assert np.isnan(mcs_fit(np.array([0.1]), 105.658, 5.0))


def test_mcs_fit_can_return_invalid_optimizer_result(monkeypatch):
    result = SimpleNamespace(x=100000.0, success=True)
    monkeypatch.setattr(
        mcs_mod.scipy.optimize, "minimize_scalar", lambda *args, **kwargs: result
    )

    assert mcs_fit(np.array([0.1]), 105.658, 5.0, return_invalid=True) == pytest.approx(
        100000.0
    )


@pytest.mark.parametrize("bounds", [(-1.0, 10.0), (10.0, 10.0), (10.0, np.inf)])
def test_mcs_fit_validates_bounds(bounds):
    with pytest.raises(ValueError, match="fit bounds"):
        mcs_fit(
            np.array([0.1]), 105.658, 5.0, lower_bound=bounds[0], upper_bound=bounds[1]
        )


def test_mcs_likelihood_and_scattering_helpers():
    """MCS likelihoods should cover resolution models and exhausted tracks."""
    theta = np.array([0.05, 0.1], dtype=np.float64)
    assert np.isfinite(mcs_nll_lar(500.0, theta, 105.658, 5.0))
    assert np.isfinite(mcs_nll_lar(500.0, theta, 105.658, 5.0, res_mixture=True))
    assert np.isinf(mcs_nll_lar(0.01, theta, 105.658, 100.0))
    with pytest.raises(AssertionError, match="angles"):
        mcs_nll_lar(500.0, np.empty(0), 105.658, 5.0)

    assert np.all(np.asarray(highland(np.array([100.0, 200.0]), 105.658, 5.0)) > 0)


def test_mcs_angle_methods_and_projections():
    """All stable angle implementations should support signed projections."""
    dirs = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float64) / np.sqrt(2.0)
    for method in range(3):
        assert mcs_angles(dirs, method).shape == (2,)
        assert mcs_angles(dirs, method, axis=2).shape == (2,)
    with pytest.raises(ValueError, match="not recognized"):
        mcs_angles(dirs, 99)

    assert mcs_angles_proj(dirs, 0).shape == (2, 3)
    assert angles_cos(dirs).shape == (2,)
    assert angles_cos(dirs, 2).shape == (2,)
    assert angles_atan2(dirs).shape == (2,)
    assert angles_atan2(dirs, 2).shape == (2,)
    assert angles_kahan(dirs, None).shape == (2,)
    assert angles_kahan(dirs, 2).shape == (2,)
    left, right = split_angles(np.array([0.2, 2.0]))
    assert left.shape == right.shape == (2,)
