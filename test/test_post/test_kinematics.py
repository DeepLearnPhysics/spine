from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import (
    ELEC_PID,
    MICHL_SHP,
    MUON_PID,
    PHOT_PID,
    PID_MASSES,
    PION_PID,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.post.reco.kinematics import (
    InteractionTopologyProcessor,
    ParticleNeutrinoLogicProcessor,
    ParticleShapeLogicProcessor,
    ParticleThresholdProcessor,
)


def make_particle(**overrides):
    data = {
        "shape": SHOWR_SHP,
        "pid": PHOT_PID,
        "pid_scores": np.array([0.1, 0.6, 0.1, 0.1, 0.05, 0.05], dtype=np.float32),
        "primary_scores": np.array([0.2, 0.8], dtype=np.float32),
        "is_primary": True,
        "ke": 0.0,
        "size": 1,
        "interaction_id": 0,
        "is_valid": True,
        "energy_deposit": 0.0,
        "energy_init": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_particle_shape_logic_enforces_pid_primary_and_michel_ke():
    particle = make_particle(shape=MICHL_SHP, pid=ELEC_PID, ke=10.0)
    processor = ParticleShapeLogicProcessor(maximum_michel_ke=5.0)

    processor.process({"reco_particles": [particle]})

    assert particle.shape == SHOWR_SHP
    assert particle.pid == ELEC_PID
    assert np.count_nonzero(particle.pid_scores[2:]) == 0
    assert particle.is_primary is True


def test_particle_threshold_updates_pid_and_primary():
    shower = make_particle(
        shape=SHOWR_SHP,
        pid_scores=np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        primary_scores=np.array([0.6, 0.4], dtype=np.float32),
    )
    track = make_particle(
        shape=TRACK_SHP,
        pid_scores=np.array([0.0, 0.0, 0.2, 0.8, 0.0, 0.0], dtype=np.float32),
        primary_scores=np.array([0.1, 0.9], dtype=np.float32),
    )
    processor = ParticleThresholdProcessor(
        shower_pid_thresholds={PHOT_PID: 0.5},
        track_pid_thresholds={PION_PID: 0.5},
        primary_threshold=0.5,
    )

    processor.process({"reco_particles": [shower, track]})

    assert shower.pid == PHOT_PID
    assert shower.is_primary is False
    assert track.pid == PION_PID
    assert track.is_primary is True


def test_particle_threshold_validates_configuration_and_complete_thresholds():
    with pytest.raises(ValueError, match="Specify one"):
        ParticleThresholdProcessor()

    particle = make_particle(
        shape=SHOWR_SHP,
        pid_scores=np.array([0.4, 0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    processor = ParticleThresholdProcessor(shower_pid_thresholds={PHOT_PID: 0.5})

    with pytest.raises(AssertionError, match="all or no"):
        processor.process({"reco_particles": [particle]})


def test_particle_neutrino_logic_promotes_or_demotes_mips():
    with pytest.raises(AssertionError, match="not recognized"):
        ParticleNeutrinoLogicProcessor(method="energy")

    pion = make_particle(
        shape=TRACK_SHP,
        pid=PION_PID,
        size=10,
        pid_scores=np.array([0.0, 0.0, 0.2, 0.8, 0.0, 0.0], dtype=np.float32),
    )
    processor = ParticleNeutrinoLogicProcessor(method="size")
    processor.process({"reco_particles": [pion]})
    assert pion.pid == MUON_PID
    assert pion.pid_scores[PION_PID] == -1.0

    muon = make_particle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        size=5,
        pid_scores=np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    )
    smaller_muon = make_particle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        size=1,
        pid_scores=np.array([0.0, 0.0, 0.8, 0.2, 0.0, 0.0], dtype=np.float32),
    )
    processor.process({"reco_particles": [muon, smaller_muon]})
    assert muon.pid == MUON_PID
    assert smaller_muon.pid == PION_PID
    assert smaller_muon.pid_scores[MUON_PID] == -1.0


def test_particle_neutrino_logic_can_select_by_score():
    pion = make_particle(
        shape=TRACK_SHP,
        pid=PION_PID,
        size=10,
        pid_scores=np.array([0.0, 0.0, 0.2, 0.8, 0.0, 0.0], dtype=np.float32),
    )
    better_pion = make_particle(
        shape=TRACK_SHP,
        pid=PION_PID,
        size=1,
        pid_scores=np.array([0.0, 0.0, 0.7, 0.3, 0.0, 0.0], dtype=np.float32),
    )
    processor = ParticleNeutrinoLogicProcessor(method="score")

    processor.process({"reco_particles": [pion, better_pion]})

    assert pion.pid == PION_PID
    assert better_pion.pid == MUON_PID

    muon = make_particle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        pid_scores=np.array([0.0, 0.0, 0.4, 0.1, 0.0, 0.0], dtype=np.float32),
    )
    better_muon = make_particle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        pid_scores=np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    )
    processor.process({"reco_particles": [muon, better_muon]})
    assert muon.pid == PION_PID
    assert better_muon.pid == MUON_PID


def test_interaction_topology_applies_ke_thresholds():
    reco_ke_particle = make_particle(pid=MUON_PID, ke=5.0)
    reco_low_particle = make_particle(pid=PION_PID, ke=1.0)
    truth_init_particle = make_particle(
        pid=MUON_PID, energy_init=PID_MASSES[MUON_PID] + 5.0
    )
    truth_unknown_particle = make_particle(pid=-1, energy_deposit=0.0)
    processor = InteractionTopologyProcessor(
        ke_thresholds={MUON_PID: 3.0, "default": 2.0},
        truth_ke_mode="energy_init",
    )

    processor.process(
        {
            "reco_interactions": [
                SimpleNamespace(particles=[reco_ke_particle, reco_low_particle])
            ],
            "truth_interactions": [
                SimpleNamespace(particles=[truth_init_particle, truth_unknown_particle])
            ],
        }
    )

    assert reco_ke_particle.is_valid is True
    assert reco_low_particle.is_valid is False
    assert truth_init_particle.is_valid is True
    assert truth_unknown_particle.is_valid is True

    scalar_processor = InteractionTopologyProcessor(ke_thresholds=2.0, run_mode="reco")
    assert scalar_processor.ke_thresholds[MUON_PID] == 2.0

    sparse_processor = InteractionTopologyProcessor(
        ke_thresholds={MUON_PID: 2.0}, run_mode="reco"
    )
    assert sparse_processor.ke_thresholds[PION_PID] == 0.0
