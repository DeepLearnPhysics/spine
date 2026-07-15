from __future__ import annotations

import numpy as np

import spine.post.reco.source as source_mod
from spine.post.reco.source import SourceAssigner


class FakeGeo:
    def get_closest_module(self, points):
        return np.arange(len(points), dtype=np.int32)

    def get_closest_tpc(self, points):
        return np.arange(len(points), dtype=np.int32) + 10


def test_source_assigner_assigns_reco_and_truth_sources(monkeypatch):
    monkeypatch.setattr(source_mod.GeoManager, "get_instance", lambda: FakeGeo())
    processor = SourceAssigner(run_mode="both", truth_point_mode="points")
    points = np.zeros((2, 3), dtype=np.float32)

    result = processor.process({"points": points, "points_label": points})

    assert np.array_equal(result["sources"], [[0, 10], [1, 11]])
    assert np.array_equal(result["sources_label"], [[0, 10], [1, 11]])


def test_source_assigner_respects_run_mode(monkeypatch):
    monkeypatch.setattr(source_mod.GeoManager, "get_instance", lambda: FakeGeo())
    processor = SourceAssigner(run_mode="reco")

    result = processor.process({"points": np.zeros((1, 3), dtype=np.float32)})

    assert set(result) == {"sources"}
