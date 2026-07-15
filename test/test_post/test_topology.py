from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import ELEC_PID, PHOT_PID, SHOWR_SHP, TRACK_SHP
from spine.post.reco import topology as topology_mod


def make_particle(**overrides):
    data = {
        "shape": TRACK_SHP,
        "pid": ELEC_PID,
        "is_primary": True,
        "points": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "depositions": np.ones(3, dtype=np.float32),
        "start_point": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "end_point": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        "start_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "end_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_particle_dedx_processor_validates_mode_and_uses_default(monkeypatch):
    with pytest.raises(AssertionError, match="not recognized"):
        topology_mod.ParticleDEDXProcessor(mode="cone")

    monkeypatch.setattr(topology_mod, "cluster_dedx", lambda *args, **kwargs: 1.5)
    particle = make_particle(
        points=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )
    skipped = make_particle(is_primary=False)
    processor = topology_mod.ParticleDEDXProcessor(mode="default")

    processor.process({"reco_particles": [particle, skipped]})

    assert particle.start_dedx == 1.5
    assert particle.end_dedx == 1.5
    assert not hasattr(skipped, "start_dedx")


def test_particle_dedx_processor_uses_direction_and_skips_shower_end(monkeypatch):
    monkeypatch.setattr(
        topology_mod, "cluster_dedx_dir", lambda *args, **kwargs: (2.5, None)
    )
    particle = make_particle(shape=SHOWR_SHP, pid=PHOT_PID)
    processor = topology_mod.ParticleDEDXProcessor(mode="direction")

    processor.process({"reco_particles": [particle]})

    assert particle.start_dedx == 2.5
    assert not hasattr(particle, "end_dedx")

    skipped = make_particle(pid=99)
    processor.process({"reco_particles": [skipped]})
    assert not hasattr(skipped, "start_dedx")


def test_particle_start_straightness_processor_branches():
    with pytest.raises(AssertionError, match="PCA components"):
        topology_mod.ParticleStartStraightnessProcessor(n_components=4)

    particle = make_particle()
    secondary = make_particle(is_primary=False)
    skipped = make_particle(pid=99)
    processor = topology_mod.ParticleStartStraightnessProcessor(radius=3.0)

    processor.process({"reco_particles": [particle, secondary, skipped]})

    assert particle.start_straightness == pytest.approx(1.0)
    assert not hasattr(secondary, "start_straightness")
    assert not hasattr(skipped, "start_straightness")

    sparse = make_particle(points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert processor.get_start_straightness(sparse) == -1.0

    flat = make_particle(points=np.zeros((3, 3), dtype=np.float32))
    assert processor.get_start_straightness(flat) == 0.0


@pytest.mark.filterwarnings(
    "ignore:An input array is constant:scipy.stats.ConstantInputWarning"
)
def test_particle_spread_processor_validates_mode_and_processes_interactions():
    with pytest.raises(AssertionError, match="not recognized"):
        topology_mod.ParticleSpreadProcessor(start_mode="centroid")

    particle = make_particle(
        points=np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [4.0, 0.5, 0.0],
                [8.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    skipped = make_particle(is_primary=False)
    skipped_pid = make_particle(pid=99)
    inter = SimpleNamespace(
        vertex=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        particles=[particle, skipped, skipped_pid],
    )
    processor = topology_mod.ParticleSpreadProcessor(
        start_mode="vertex", use_start_dir=True
    )

    processor.process({"reco_interactions": [inter]})

    assert particle.directional_spread >= 0.0
    assert np.isfinite(particle.axial_spread)
    assert not hasattr(skipped, "directional_spread")
    assert not hasattr(skipped_pid, "directional_spread")

    particle = make_particle()
    inter = SimpleNamespace(vertex=np.zeros(3, dtype=np.float32), particles=[particle])
    processor = topology_mod.ParticleSpreadProcessor(start_mode="start_point")
    processor.process({"reco_interactions": [inter]})
    assert hasattr(particle, "directional_spread")


def test_particle_spread_processor_helper_degenerate_cases():
    processor = topology_mod.ParticleSpreadProcessor(
        start_mode="start_point", use_start_dir=True
    )
    points = np.zeros((2, 3), dtype=np.float32)
    ref_point = np.zeros(3, dtype=np.float32)

    assert processor.get_dir_spread(points, ref_point) == -1.0
    assert processor.get_axial_spread(points, np.array([1.0, 0.0, 0.0]), ref_point) == (
        -np.inf
    )

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    processor = topology_mod.ParticleSpreadProcessor(
        start_mode="start_point", use_start_dir=True
    )
    assert np.isfinite(
        processor.get_axial_spread(points, np.array([1.0, 0.0, 0.0]), ref_point)
    )

    processor = topology_mod.ParticleSpreadProcessor(start_mode="start_point")
    assert np.isfinite(
        processor.get_axial_spread(points, np.array([-1.0, 0.0, 0.0]), ref_point)
    )
