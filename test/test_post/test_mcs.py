import numpy as np

from spine.constants import MUON_PID, TRACK_SHP
from spine.data.out import RecoParticle
from spine.post.reco.mcs import MCSEnergyProcessor


def test_mcs_skips_one_point_track():
    processor = MCSEnergyProcessor(run_mode="reco")
    particle = RecoParticle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        start_point=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    processor.process({"reco_particles": [particle]})

    assert np.isnan(particle.mcs_ke)
