from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from spine.constants import COORD_COLS, PPN_SHAPE_COL, TRACK_SHP
from spine.post.reco import ppn as ppn_mod


class FakePPNPredictor:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.data = None

    def __call__(self, **data):
        self.data = data
        prediction = np.zeros((2, PPN_SHAPE_COL + 1), dtype=np.float32)
        prediction[0, COORD_COLS] = [0.0, 0.0, 0.0]
        prediction[0, PPN_SHAPE_COL] = TRACK_SHP
        prediction[1, COORD_COLS] = [10.0, 0.0, 0.0]
        prediction[1, PPN_SHAPE_COL] = TRACK_SHP + 1
        return [prediction]


def test_ppn_processor_builds_candidates(monkeypatch):
    monkeypatch.setattr(ppn_mod, "PPNPredictor", FakePPNPredictor)
    processor = ppn_mod.PPNProcessor(foo="bar")

    result = processor.process(
        {
            "segmentation": np.empty((0, 1)),
            "ppn_points": np.empty((0, 1)),
            "ppn_coords": np.empty((0, 1)),
            "ppn_masks": np.empty((0, 1)),
        }
    )

    assert np.asarray(result["ppn_pred"]).shape == (2, PPN_SHAPE_COL + 1)
    assert cast(FakePPNPredictor, processor.ppn_predictor).cfg == {"foo": "bar"}


def test_ppn_processor_prefers_unique_sparse_predictions(monkeypatch):
    monkeypatch.setattr(ppn_mod, "PPNPredictor", FakePPNPredictor)
    processor = ppn_mod.PPNProcessor()
    restored = np.zeros((2, 1))
    unique = np.ones((1, 1))
    restored_endpoints = np.zeros((2, 2))
    unique_endpoints = np.ones((1, 2))

    processor.process(
        {
            "segmentation": np.empty((0, 1)),
            "ppn_points": restored,
            "ppn_points_unique": unique,
            "ppn_coords": np.empty((0, 1)),
            "ppn_masks": np.empty((0, 1)),
            "ppn_classify_endpoints": restored_endpoints,
            "ppn_classify_endpoints_unique": unique_endpoints,
        }
    )

    predictor = cast(FakePPNPredictor, processor.ppn_predictor)
    assert predictor.data is not None
    assert predictor.data["ppn_points"][0] is unique
    assert predictor.data["ppn_classify_endpoints"][0] is unique_endpoints


def test_ppn_processor_assigns_candidates_to_particles(monkeypatch):
    monkeypatch.setattr(ppn_mod, "PPNPredictor", FakePPNPredictor)
    processor = ppn_mod.PPNProcessor(
        assign_to_particles=True, restrict_shape=True, match_threshold=1.0
    )
    particle = SimpleNamespace(
        shape=TRACK_SHP,
        points=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
    )

    processor.process(
        {
            "segmentation": np.empty((0, 1)),
            "ppn_points": np.empty((0, 1)),
            "ppn_coords": np.empty((0, 1)),
            "ppn_masks": np.empty((0, 1)),
            "reco_particles": [particle],
        }
    )

    assert np.array_equal(particle.ppn_points, np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(particle.ppn_ids, np.array([0]))


def test_ppn_processor_assigns_unrestricted_candidate_ids(monkeypatch):
    monkeypatch.setattr(ppn_mod, "PPNPredictor", FakePPNPredictor)
    processor = ppn_mod.PPNProcessor(assign_to_particles=True, match_threshold=20.0)
    particle = SimpleNamespace(
        shape=TRACK_SHP,
        points=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
    )

    processor.process(
        {
            "segmentation": np.empty((0, 1)),
            "ppn_points": np.empty((0, 1)),
            "ppn_coords": np.empty((0, 1)),
            "ppn_masks": np.empty((0, 1)),
            "reco_particles": [particle],
        }
    )

    assert np.array_equal(particle.ppn_ids, np.array([0, 1]))
