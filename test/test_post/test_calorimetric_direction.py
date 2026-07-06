"""Tests for calorimetric interaction direction reconstruction."""

from types import SimpleNamespace

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


def test_calorimetric_direction_skips_degenerate_inputs():
    processor = CalorimetricDirectionProcessor(run_mode="reco")
    data = {
        "points": np.zeros((1, 3), dtype=np.float32),
        "depositions": np.ones(1, dtype=np.float32),
    }

    inter = SimpleNamespace(
        is_truth=False,
        vertex=None,
        dir=None,
        particles=[RecoParticle(index=np.array([0]))],
    )
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None

    inter = SimpleNamespace(
        is_truth=False,
        vertex=np.zeros(3),
        dir=None,
        particles=[],
    )
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None

    inter = SimpleNamespace(
        is_truth=False,
        vertex=np.zeros(3),
        dir=None,
        particles=[SimpleNamespace(size=0, index=np.empty(0, dtype=np.int64))],
    )
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None

    inter = SimpleNamespace(
        is_truth=False,
        vertex=np.zeros(3),
        dir=None,
        particles=[SimpleNamespace(size=1, index=np.empty(0, dtype=np.int64))],
    )
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None

    inter = SimpleNamespace(
        is_truth=False,
        vertex=np.zeros(3),
        dir=None,
        particles=[RecoParticle(index=np.array([0]))],
    )
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None

    data["points"] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    data["depositions"] = np.zeros(1, dtype=np.float32)
    processor._reconstruct_direction(inter, data, "points")
    assert inter.dir is None
