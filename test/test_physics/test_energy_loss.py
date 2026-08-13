"""Tests for packaged continuous-slowing-down approximation tables."""

import numpy as np
import pytest

from spine.constants import (
    KAON_PID,
    LAR_DE_X0,
    LAR_DE_X1,
    MUON_MASS,
    MUON_PID,
    PION_PID,
    PROT_PID,
)
from spine.physics.energy_loss import (
    W_max,
    bethe_bloch_lar,
    bethe_bloch_mpv_lar,
    csda_ke_lar,
    csda_range_lar,
    csda_table_spline,
    delta_lar,
    inv_bethe_bloch_lar,
    le_corr_lar,
    step_energy_loss_lar,
)


@pytest.mark.parametrize("pid", [MUON_PID, PION_PID, KAON_PID, PROT_PID])
def test_csda_tables_are_packaged_and_loadable(pid):
    """Every supported particle should resolve its model-owned data table."""
    spline = csda_table_spline(pid)
    value = spline(10.0)
    assert np.isfinite(value)
    assert value > 0.0


def test_csda_table_rejects_unsupported_particle():
    """Unavailable particle hypotheses should fail with useful context."""
    with pytest.raises(ValueError, match="not available"):
        csda_table_spline(-1)


def test_csda_table_values_and_numerical_range_inverse():
    """Tables and numerical CSDA integration should expose both supported values."""
    assert csda_table_spline(MUON_PID, value="dE/dx")(10.0) > 0.0
    with pytest.raises(AssertionError, match="kinetic energy"):
        csda_table_spline(MUON_PID, value="bad")

    assert csda_range_lar(0.0, MUON_MASS) == 0.0
    track_range = csda_range_lar(10.0, MUON_MASS)
    assert track_range > 0.0
    assert csda_ke_lar(track_range, MUON_MASS, T_max=100.0) == pytest.approx(
        10.0, rel=1e-2
    )


def test_energy_loss_formula_branches():
    """Stopping-power helpers should cover bounded stepping and material corrections."""
    with pytest.raises(AssertionError, match="positive"):
        step_energy_loss_lar(0.0, MUON_MASS, 1.0)
    limited = step_energy_loss_lar(100.0, MUON_MASS, 1.0, num_steps=2)
    assert len(limited) == 3
    stopped = step_energy_loss_lar(1.0, MUON_MASS, 100.0, num_steps=10)
    assert stopped[-1] == 0.0

    loss = bethe_bloch_lar(100.0, MUON_MASS)
    assert loss < 0.0
    assert inv_bethe_bloch_lar(100.0, MUON_MASS) == pytest.approx(1.0 / loss)
    assert bethe_bloch_mpv_lar(100.0, MUON_MASS, 1.0) < 0.0
    gamma = 1.0 + 100.0 / MUON_MASS
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    assert W_max(beta, gamma, MUON_MASS) > 0.0
    assert np.isfinite(le_corr_lar(beta))

    for exponent in (LAR_DE_X0 - 1.0, (LAR_DE_X0 + LAR_DE_X1) / 2, LAR_DE_X1 + 1.0):
        assert np.isfinite(delta_lar(10**exponent))
