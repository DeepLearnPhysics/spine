from types import SimpleNamespace

import numpy as np
import pytest

import spine.utils.mcs as mcs_mod
from spine.utils.mcs import mcs_fit


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
