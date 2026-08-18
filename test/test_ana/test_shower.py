"""Tests for the shower-start dE/dx diagnostic analyzer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spine.ana.diag.shower as shower_mod
from spine.ana.diag.shower import ShowerStartDEdxAna
from spine.constants import ELEC_PID, SHOWR_SHP, TRACK_SHP


@pytest.fixture(autouse=True)
def _capture_writer_initialization(monkeypatch):
    """Avoid creating CSV files during analyzer tests."""
    monkeypatch.setattr(
        ShowerStartDEdxAna, "initialize_writer", lambda self, name: None
    )


def make_shower(*, truth=False, **overrides):
    """Build a minimal reconstructed or truth shower-like object."""
    values = {
        "id": 7,
        "is_truth": truth,
        "shape": SHOWR_SHP,
        "pid": ELEC_PID,
        "is_primary": True,
        "start_point": np.zeros(3, dtype=np.float32),
        "start_dir": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        "points": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "depositions": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "points_g4": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        "depositions_g4": np.array([4.0, 2.0], dtype=np.float32),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shower_start_dedx_validates_and_normalizes_configuration():
    """Scan parameters should be normalized and invalid values rejected."""
    ana = ShowerStartDEdxAna(
        radius=[1, 2.5], mode=["default", "direction"], run_mode="reco"
    )

    assert ana.radii == (1.0, 2.5)
    assert ana.modes == ("default", "direction")
    assert ana.obj_keys == ["reco_particles"]

    with pytest.raises(ValueError, match="At least one dE/dx radius"):
        ShowerStartDEdxAna(radius=[])
    with pytest.raises(ValueError, match="finite and positive"):
        ShowerStartDEdxAna(radius=[1.0, np.nan])
    with pytest.raises(ValueError, match="At least one dE/dx mode"):
        ShowerStartDEdxAna(radius=1.0, mode=[])
    with pytest.raises(ValueError, match="not recognized"):
        ShowerStartDEdxAna(radius=1.0, mode="cone")
    with pytest.raises(ValueError, match="does not support interactions"):
        ShowerStartDEdxAna(radius=1.0, obj_type="interaction")


def test_shower_start_dedx_records_mode_radius_cross_product(monkeypatch):
    """Each shower should produce one row per requested mode and radius."""
    rows = []
    default_calls = []
    direction_calls = []
    monkeypatch.setattr(
        ShowerStartDEdxAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )

    def fake_default(points, values, start, **kwargs):
        max_dist, anchor = kwargs["max_dist"], kwargs["anchor"]
        default_calls.append((points, values, start, max_dist, anchor))
        return max_dist

    def fake_direction(points, values, start, direction, **kwargs):
        max_dist, anchor = kwargs["max_dist"], kwargs["anchor"]
        direction_calls.append((points, values, start, direction, max_dist, anchor))
        return 2.0 * max_dist, 0.0, 0.0, 0.0, len(points)

    monkeypatch.setattr(shower_mod, "cluster_dedx", fake_default)
    monkeypatch.setattr(shower_mod, "cluster_dedx_dir", fake_direction)

    shower = make_shower()
    track = make_shower(shape=TRACK_SHP)
    ana = ShowerStartDEdxAna(
        radius=[1.0, 3.0],
        mode=["default", "direction"],
        anchor=True,
        run_mode="reco",
    )
    ana.process({"reco_particles": [shower, track]})

    assert len(rows) == 4
    assert {row[1]["dedx"] for row in rows} == {1.0, 2.0, 3.0, 6.0}
    assert all(name == "reco_particles" for name, _ in rows)
    assert all(row["object_id"] == 7 for _, row in rows)
    assert all(row["anchor"] for _, row in rows)
    assert [call[3] for call in default_calls] == [1.0, 3.0]
    assert [call[4] for call in direction_calls] == [1.0, 3.0]
    np.testing.assert_allclose(direction_calls[0][3], [1.0, 0.0, 0.0])


def test_shower_start_dedx_uses_configured_truth_representation(monkeypatch):
    """Truth measurements should use the selected point/deposition attributes."""
    calls = []
    rows = []
    monkeypatch.setattr(
        shower_mod,
        "cluster_dedx",
        lambda points, values, *args, **kwargs: calls.append((points, values)) or 4.0,
    )
    monkeypatch.setattr(
        ShowerStartDEdxAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    shower = make_shower(truth=True)
    ana = ShowerStartDEdxAna(
        radius=2.0,
        run_mode="truth",
        truth_point_mode="points_g4",
        truth_dep_mode="depositions_g4",
    )

    ana.process({"truth_particles": [shower]})

    np.testing.assert_array_equal(calls[0][0], shower.points_g4)
    np.testing.assert_array_equal(calls[0][1], shower.depositions_g4)
    assert rows[0][0] == "truth_particles"
    assert rows[0][1]["dedx"] == 4.0


def test_shower_start_dedx_uses_shared_cluster_kernels():
    """Both supported modes should produce the expected straight-line dE/dx."""
    shower = make_shower()
    ana = ShowerStartDEdxAna(radius=2.0, run_mode="reco")

    default = ana.local_dedx(shower, shower.points, shower.depositions, 2.0, "default")
    direction = ana.local_dedx(
        shower, shower.points, shower.depositions, 2.0, "direction"
    )

    assert default == pytest.approx(3.0)
    assert direction == pytest.approx(3.0)


def test_shower_start_dedx_handles_unusable_or_inconsistent_inputs():
    """Undefined geometries are recorded as NaN; inconsistent arrays fail."""
    ana = ShowerStartDEdxAna(radius=2.0, mode="direction", run_mode="reco")
    points = np.zeros((1, 3), dtype=np.float32)
    values = np.ones(1, dtype=np.float32)
    shower = make_shower()

    assert np.isnan(ana.local_dedx(shower, points, values, 2.0, "direction"))

    bad_start = make_shower(start_point=np.full(3, np.nan, dtype=np.float32))
    assert np.isnan(
        ana.local_dedx(
            bad_start, bad_start.points, bad_start.depositions, 2.0, "direction"
        )
    )

    bad_direction = make_shower(start_dir=np.zeros(3, dtype=np.float32))
    assert np.isnan(
        ana.local_dedx(
            bad_direction,
            bad_direction.points,
            bad_direction.depositions,
            2.0,
            "direction",
        )
    )
    with pytest.raises(ValueError, match="Unsupported dE/dx mode"):
        ana.local_dedx(shower, shower.points, shower.depositions, 2.0, "unsupported")

    with pytest.raises(ValueError, match="matching lengths"):
        ana.process(
            {"reco_particles": [make_shower(depositions=np.ones(2, dtype=np.float32))]}
        )
