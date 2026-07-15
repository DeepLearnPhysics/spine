from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.post.crt import match as match_mod


class FakeCRT:
    def __init__(self, normal):
        self.normal = np.asarray(normal, dtype=np.float32)

    def get_plane(self, position, plane):
        return self


class FakeGeo:
    def __init__(self, has_crt=True, multi_tpc=False, drift_sign=1):
        self.crt = FakeCRT([1.0, 0.0, 0.0]) if has_crt else None
        self.multi_tpc = multi_tpc
        chamber = SimpleNamespace(
            drift_sign=drift_sign,
            drift_axis=0,
            anode_pos=10.0,
            cathode_pos=0.0,
        )
        self.tpc = [[chamber]]

    def get_contributors(self, sources):
        if self.multi_tpc and len(sources) > 1:
            return np.array([0, 0]), np.array([0, 1])
        return np.array([0]), np.array([0])


def make_particle(**overrides):
    data = {
        "id": 0,
        "shape": TRACK_SHP,
        "size": 3,
        "points": np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "sources": np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int64),
        "start_point": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "end_point": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "start_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "end_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_crt_matcher_validates_configuration(monkeypatch):
    monkeypatch.setattr(match_mod.GeoManager, "get_instance", lambda: FakeGeo())

    with pytest.raises(ValueError, match="not recognized"):
        match_mod.CRTMatcher(driftv=0.1, match_method="closest")

    with pytest.raises(AssertionError, match="match_distance"):
        match_mod.CRTMatcher(driftv=0.1, match_method="threshold")


def test_crt_matcher_restricts_objects_and_line_plane_intercepts(monkeypatch):
    monkeypatch.setattr(match_mod.GeoManager, "get_instance", lambda: FakeGeo())
    matcher = match_mod.CRTMatcher(
        driftv=0.1,
        match_method="threshold",
        match_distance=1.0,
        time_window=(0.0, 10.0),
        min_part_size=2,
    )
    track = make_particle(size=3)
    small_track = make_particle(size=1)
    shower = make_particle(shape=SHOWR_SHP)
    early_hit = SimpleNamespace(time=-1.0)
    good_hit = SimpleNamespace(time=1.0)

    particles, hits = matcher.restrict_objects(
        [track, small_track, shower], [early_hit, good_hit]
    )

    assert particles == [track]
    assert hits == [good_hit]
    assert np.array_equal(
        matcher.line_plane_intercept(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.ones(3),
            np.array([1.0, 0.0, 0.0]),
        ),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.all(
        np.isneginf(
            matcher.line_plane_intercept(
                np.zeros(3),
                np.array([0.0, 1.0, 0.0]),
                np.ones(3),
                np.array([1.0, 0.0, 0.0]),
            )
        )
    )


def test_crt_matcher_finds_matches(monkeypatch):
    monkeypatch.setattr(match_mod.GeoManager, "get_instance", lambda: FakeGeo())
    matcher = match_mod.CRTMatcher(
        driftv=0.1, match_method="threshold", match_distance=0.1
    )
    particle = make_particle()
    hit = SimpleNamespace(
        id=0,
        time=0.0,
        center=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        plane=0,
    )

    matches = matcher.get_matches([particle], [hit])

    assert matches[0][0] is particle
    assert matches[0][1] is hit
    assert matches[0][2] == pytest.approx(0.0)


def test_crt_matcher_handles_negative_drift_sign(monkeypatch):
    monkeypatch.setattr(
        match_mod.GeoManager, "get_instance", lambda: FakeGeo(drift_sign=-1)
    )
    matcher = match_mod.CRTMatcher(
        driftv=0.1, match_method="threshold", match_distance=0.1
    )
    particle = make_particle(
        points=np.array(
            [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        start_point=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.array([2.0, 0.0, 0.0], dtype=np.float32),
    )
    hit = SimpleNamespace(
        id=0,
        time=0.0,
        center=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        plane=0,
    )

    assert matcher.get_matches([particle], [hit]) == []


def test_crt_matcher_handles_empty_and_invalid_geometry(monkeypatch):
    monkeypatch.setattr(match_mod.GeoManager, "get_instance", lambda: FakeGeo())
    matcher = match_mod.CRTMatcher(
        driftv=0.1, match_method="threshold", match_distance=0.1
    )
    assert (
        matcher.get_matches([make_particle(shape=SHOWR_SHP)], [SimpleNamespace()]) == []
    )

    monkeypatch.setattr(match_mod.GeoManager, "get_instance", lambda: FakeGeo(False))
    matcher = match_mod.CRTMatcher(
        driftv=0.1, match_method="threshold", match_distance=0.1
    )
    with pytest.raises(AssertionError, match="crt"):
        matcher.get_matches([make_particle()], [SimpleNamespace()])


def test_crt_matcher_handles_multi_tpc_particle(monkeypatch):
    monkeypatch.setattr(
        match_mod.GeoManager, "get_instance", lambda: FakeGeo(multi_tpc=True)
    )
    matcher = match_mod.CRTMatcher(
        driftv=0.1, match_method="threshold", match_distance=0.1
    )
    particle = make_particle(sources=np.array([[0, 0], [0, 1], [0, 1]], dtype=np.int64))
    hit = SimpleNamespace(
        id=0,
        time=0.0,
        center=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        plane=0,
    )

    assert len(matcher.get_matches([particle], [hit])) == 1
