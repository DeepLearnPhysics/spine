from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from spine.post.crt import crt_matching as crt_matching_mod


class FakeMatcher:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_matches(self, particles, crthits):
        return [(particles[0], crthits[0], 0.25)]


def make_particle():
    particle = SimpleNamespace(
        id=0,
        units="cm",
        is_crt_matched=True,
        crt_ids=np.array([99], dtype=np.int32),
        crt_times=np.array([99.0], dtype=np.float32),
        crt_scores=np.array([99.0], dtype=np.float32),
    )

    def reset_crt_match():
        particle.is_crt_matched = False
        particle.crt_ids = np.empty(0, dtype=np.int32)
        particle.crt_times = np.empty(0, dtype=np.float32)
        particle.crt_scores = np.empty(0, dtype=np.float32)

    particle.reset_crt_match = reset_crt_match
    return particle


def test_crt_match_processor_updates_particle_matches(monkeypatch):
    monkeypatch.setattr(crt_matching_mod, "CRTMatcher", FakeMatcher)
    processor = crt_matching_mod.CRTMatchProcessor("crthits", driftv=0.1)
    particle = make_particle()
    crthit = SimpleNamespace(id=5, time=12.0)

    processor.process({"reco_particles": [particle], "crthits": [crthit]})

    assert particle.is_crt_matched is True
    assert np.array_equal(particle.crt_ids, np.array([5], dtype=np.int32))
    assert np.array_equal(particle.crt_times, np.array([12.0], dtype=np.float32))
    assert np.array_equal(particle.crt_scores, np.array([0.25], dtype=np.float32))
    assert cast(FakeMatcher, processor.matcher).kwargs == {"driftv": 0.1}


def test_crt_match_processor_skips_empty_inputs(monkeypatch):
    monkeypatch.setattr(crt_matching_mod, "CRTMatcher", FakeMatcher)
    processor = crt_matching_mod.CRTMatchProcessor("crthits", driftv=0.1)
    particle = make_particle()

    assert processor.process({"reco_particles": [particle], "crthits": []}) is None
    assert particle.is_crt_matched is True

    processor.process({"reco_particles": [], "crthits": [SimpleNamespace()]})
