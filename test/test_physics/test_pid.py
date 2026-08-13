"""Tests for template-based particle identification."""

import numpy as np
import pytest

import spine.physics.pid as pid_module
from spine.constants import MUON_PID, PROT_PID
from spine.physics.pid import TemplateParticleIdentifier


def make_identifier(monkeypatch, **kwargs):
    """Build an identifier without depending on table file interpolation."""
    monkeypatch.setattr(
        pid_module, "csda_table_spline", lambda *args, **kw: lambda x: x + 1.0
    )
    return TemplateParticleIdentifier(include_pids=(MUON_PID, PROT_PID), **kwargs)


def test_pid_validation_expected_values_and_chi2(monkeypatch):
    """Table and theoretical templates should supply finite chi-square values."""
    identifier = make_identifier(monkeypatch, optimize_orient=False)
    rrs = np.array([1.0, 2.0])
    np.testing.assert_allclose(identifier.expected_dedxs(rrs, MUON_PID), rrs + 1.0)
    assert identifier.chi2(np.array([2.0, 3.0]), rrs, MUON_PID) == 0.0

    with pytest.raises(AssertionError, match="MPV"):
        make_identifier(monkeypatch, use_table=True, use_mpv=True)

    monkeypatch.setattr(pid_module, "csda_ke_lar", lambda rr, mass: rr)
    monkeypatch.setattr(pid_module, "bethe_bloch_lar", lambda energy, mass: -2.0)
    theoretical = make_identifier(monkeypatch, use_table=False, optimize_orient=False)
    np.testing.assert_allclose(theoretical.expected_dedxs(rrs, MUON_PID), 2.0)

    monkeypatch.setattr(pid_module, "bethe_bloch_mpv_lar", lambda energy, mass, x: -3.0)
    mpv = make_identifier(
        monkeypatch, use_table=False, use_mpv=True, optimize_orient=False
    )
    np.testing.assert_allclose(mpv.expected_dedxs(rrs, MUON_PID), 3.0)


def test_pid_track_dispatch_orientation_and_range_fit(monkeypatch):
    """PID should handle empty measurements, both orientations, and RR fitting."""
    identifier = make_identifier(monkeypatch)
    points = np.zeros((2, 3))
    depositions = np.ones(2)

    with pytest.raises(AssertionError, match="start_point"):
        identifier(points, depositions, np.zeros(3))

    monkeypatch.setattr(
        pid_module,
        "get_track_segment_dedxs",
        lambda *args, **kwargs: (
            np.array([-1.0]),
            None,
            np.array([1.0]),
            None,
            None,
            None,
        ),
    )
    best, scores = identifier(points, depositions, np.zeros(3), np.ones(3))
    assert best == -1
    np.testing.assert_array_equal(scores, -1.0)

    monkeypatch.setattr(
        pid_module,
        "get_track_segment_dedxs",
        lambda *args, **kwargs: (
            np.array([2.0, 3.0]),
            None,
            np.array([1.0, 2.0]),
            None,
            None,
            None,
        ),
    )
    best, scores = identifier(points, depositions, np.zeros(3), np.ones(3))
    assert best == MUON_PID
    assert np.all(scores == 0.0)

    fitting = make_identifier(
        monkeypatch, optimize_rr=True, optimize_orient=False, max_rr=5.0
    )
    offset, score = fitting.minimize_rr(
        np.array([2.0, 3.0]), np.array([1.0, 2.0]), MUON_PID
    )
    assert 0.0 <= offset <= 5.0
    assert score >= 0.0
    best, scores = fitting(points, depositions, np.zeros(3))
    assert best == MUON_PID
    assert np.all(scores >= 0.0)
