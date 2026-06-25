"""Tests for calorimetric interaction direction reconstruction."""

import numpy as np

from spine.data.out import (
    RecoInteraction,
    RecoParticle,
    TruthInteraction,
    TruthParticle,
)
from spine.post.reco.calorimetric_direction import CalorimetricDirectionProcessor


def test_calorimetric_direction_sets_reco_and_truth_direction_attributes():
    processor = CalorimetricDirectionProcessor(run_mode="both")
    reco_inter = RecoInteraction(
        vertex=np.zeros(3, dtype=np.float32),
        particles=[RecoParticle(index=np.array([0], dtype=np.int32))],
    )
    truth_inter = TruthInteraction(
        momentum=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        reco_vertex=np.zeros(3, dtype=np.float32),
        particles=[TruthParticle(index=np.array([1], dtype=np.int32))],
    )

    processor.process(
        {
            "points": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "points_label": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),
            "depositions": np.ones(2, dtype=np.float32),
            "reco_interactions": [reco_inter],
            "truth_interactions": [truth_inter],
        }
    )

    np.testing.assert_allclose(reco_inter.dir, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(reco_inter.reco_dir, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(truth_inter.dir, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(truth_inter.reco_dir, [0.0, 1.0, 0.0])
