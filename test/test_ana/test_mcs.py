from __future__ import annotations

import numpy as np
import pytest

import spine.ana.calib.mcs as mcs_mod
from spine.ana.calib.mcs import MCSCalibAna
from spine.constants import MUON_PID


class FakeParticle:
    is_truth = True
    pid = MUON_PID
    ke = 250.0
    t = 5.0
    start_point = np.zeros(3, dtype=np.float32)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32
    )


@pytest.fixture(autouse=True)
def _capture_writers(monkeypatch):
    monkeypatch.setattr(MCSCalibAna, "initialize_writer", lambda self, name: None)


def test_mcs_calib_ana_validates_configuration():
    with pytest.raises(ValueError, match="Angular reconstruction method"):
        MCSCalibAna(min_ke=100.0, segment_length=10.0, angle_method="bad")

    with pytest.raises(ValueError, match="reconstructed objects"):
        MCSCalibAna(
            min_ke=100.0,
            segment_length=10.0,
            time_window=(0.0, 1.0),
            run_mode="both",
        )

    with pytest.raises(ValueError, match="two scalars"):
        MCSCalibAna(min_ke=100.0, segment_length=10.0, time_window=1.0)


def test_mcs_calib_ana_process_writes_angle_rows(monkeypatch):
    rows = []
    monkeypatch.setattr(
        MCSCalibAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    monkeypatch.setattr(mcs_mod, "mcs_angles", lambda dirs, method: np.array([0.5]))
    monkeypatch.setattr(
        mcs_mod, "mcs_angles_proj", lambda dirs, method: np.array([[0.1, 0.2, 0.3]])
    )

    ana = MCSCalibAna(min_ke=100.0, segment_length=[10.0], time_window=(0.0, 10.0))
    ana.get_track_segments = lambda points, length, start: (
        [np.array([0, 1]), np.array([2])],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([2.0, 4.0]),
    )

    ana.process({"truth_particles": [FakeParticle()]})

    assert rows == [
        (
            "truth_mcs_sl10.0",
            {
                "ke": 250.0,
                "time": 5.0,
                "dir_x": 0.5,
                "dir_y": 0.5,
                "dir_z": 0.0,
                "angle_yz": 0.1,
                "angle_xz": 0.2,
                "angle_xy": 0.3,
                "angle": 0.5,
                "min_count": 1,
                "distance": 3.0,
            },
        )
    ]


def test_mcs_calib_ana_skips_unusable_particles_and_empty_segments(monkeypatch):
    rows = []
    monkeypatch.setattr(
        MCSCalibAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    ana = MCSCalibAna(min_ke=100.0, segment_length=10.0, time_window=(0.0, 10.0))
    ana.get_track_segments = lambda points, length, start: ([], np.empty((0, 3)), [])

    low_ke = FakeParticle()
    low_ke.ke = 50.0
    out_of_time = FakeParticle()
    out_of_time.t = 20.0
    no_segments = FakeParticle()

    ana.process({"truth_particles": [low_ke, out_of_time, no_segments]})

    assert rows == []
