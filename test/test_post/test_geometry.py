from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spine.post.reco import geometry as geometry_mod


class FakeGeo:
    def __init__(self):
        self.definitions = []
        self.checks = []

    def define_containment_volumes(
        self, margin, cathode_margin=None, mode="module", include_limits=True
    ):
        definition = {
            "margin": margin,
            "cathode_margin": cathode_margin,
            "mode": mode,
            "include_limits": include_limits,
        }
        self.definitions.append(definition)
        return definition

    def check_containment(
        self, definition, points, sources=None, allow_multi_module=False
    ):
        self.checks.append(
            {
                "definition": definition,
                "points": points,
                "sources": sources,
                "allow_multi_module": allow_multi_module,
            }
        )
        return bool(np.all(points >= 0.0))


def test_containment_processor_meta_mode_particles_and_interactions():
    processor = geometry_mod.ContainmentProcessor(
        margin=1.0,
        mode="meta",
        run_mode="reco",
        obj_type=("particle", "interaction"),
        exclude_pids=[4],
        min_particle_sizes=cast(Any, {2: 5, "default": 0}),
    )
    contained = SimpleNamespace(
        is_truth=False,
        units="cm",
        points=np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
        sources=np.empty((1, 2), dtype=np.int64),
        pid=2,
        size=10,
        is_contained=False,
    )
    empty = SimpleNamespace(
        is_truth=False,
        units="cm",
        points=np.empty((0, 3), dtype=np.float32),
        sources=np.empty((0, 2), dtype=np.int64),
        pid=2,
        size=0,
        is_contained=False,
    )
    outside_excluded = SimpleNamespace(pid=4, size=10, is_contained=False)
    outside_small = SimpleNamespace(pid=2, size=1, is_contained=False)
    outside_large = SimpleNamespace(pid=2, size=10, is_contained=False)
    data = {
        "meta": SimpleNamespace(
            lower=np.zeros(3, dtype=np.float32),
            upper=np.full(3, 10.0, dtype=np.float32),
        ),
        "reco_particles": [contained, empty],
        "reco_interactions": [
            SimpleNamespace(particles=[contained]),
            SimpleNamespace(particles=[outside_excluded]),
            SimpleNamespace(particles=[outside_small]),
            SimpleNamespace(particles=[outside_large]),
        ],
    }

    processor.process(data)

    assert bool(contained.is_contained) is True
    assert bool(empty.is_contained) is True
    assert [inter.is_contained for inter in data["reco_interactions"]] == [
        True,
        True,
        True,
        False,
    ]


def test_containment_processor_uses_geometry_and_truth_g4(monkeypatch):
    fake_geo = FakeGeo()
    monkeypatch.setattr(geometry_mod.GeoManager, "get_instance", lambda: fake_geo)
    processor = geometry_mod.ContainmentProcessor(
        margin=1.0,
        mode="module",
        run_mode="truth",
        obj_type="particle",
        truth_point_mode="points_g4",
        allow_multi_module=True,
    )
    particle = SimpleNamespace(
        is_truth=True,
        units="cm",
        points_g4=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        sources_g4=np.array([[0, 0]], dtype=np.int64),
    )

    processor.process({"truth_particles": [particle]})

    assert particle.is_contained is True
    assert len(fake_geo.definitions) == 2
    assert fake_geo.checks[0]["sources"] is None
    assert fake_geo.checks[0]["allow_multi_module"] is True


def test_containment_processor_uses_reco_geometry_sources(monkeypatch):
    fake_geo = FakeGeo()
    monkeypatch.setattr(geometry_mod.GeoManager, "get_instance", lambda: fake_geo)
    processor = geometry_mod.ContainmentProcessor(
        margin=1.0,
        mode="module",
        run_mode="reco",
        obj_type="particle",
        min_particle_sizes=cast(Any, {}),
    )
    particle = SimpleNamespace(
        is_truth=False,
        units="cm",
        points=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        sources=np.array([[0, 0]], dtype=np.int64),
    )

    processor.process({"reco_particles": [particle]})

    assert particle.is_contained is True
    assert np.array_equal(fake_geo.checks[0]["sources"], particle.sources)
    assert processor.min_particle_sizes[0] == 0


def test_containment_processor_validates_logical_mode(monkeypatch):
    monkeypatch.setattr(geometry_mod.GeoManager, "get_instance", lambda: FakeGeo())

    with pytest.raises(AssertionError, match="logical containment"):
        geometry_mod.ContainmentProcessor(margin=1.0, mode="module", logical=True)


def test_fiducial_processor_meta_and_geometry_modes(monkeypatch):
    meta_processor = geometry_mod.FiducialProcessor(
        margin=1.0, mode="meta", run_mode="both", truth_vertex_mode="reco_vertex"
    )
    reco_inter = SimpleNamespace(
        is_truth=False,
        units="cm",
        vertex=np.array([2.0, 2.0, 2.0], dtype=np.float32),
    )
    truth_inter = SimpleNamespace(
        is_truth=True,
        units="cm",
        vertex=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        reco_vertex=np.array([9.5, 2.0, 2.0], dtype=np.float32),
    )

    meta_processor.process(
        {
            "meta": SimpleNamespace(
                lower=np.zeros(3, dtype=np.float32),
                upper=np.full(3, 10.0, dtype=np.float32),
            ),
            "reco_interactions": [reco_inter],
            "truth_interactions": [truth_inter],
        }
    )

    assert bool(reco_inter.is_fiducial) is True
    assert bool(truth_inter.is_fiducial) is False

    fake_geo = FakeGeo()
    monkeypatch.setattr(geometry_mod.GeoManager, "get_instance", lambda: fake_geo)
    geo_processor = geometry_mod.FiducialProcessor(
        margin=1.0, mode="module", run_mode="reco"
    )
    geo_processor.process({"reco_interactions": [reco_inter]})

    assert reco_inter.is_fiducial is True
    assert fake_geo.definitions[0]["include_limits"] is False


def test_fiducial_processor_validates_truth_vertex_mode():
    with pytest.raises(AssertionError, match="truth_vertex_mode"):
        geometry_mod.FiducialProcessor(
            margin=1.0, mode="meta", truth_vertex_mode="truth_start"
        )
