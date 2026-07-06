from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.post.reco import cathode_cross as cathode_mod


class FakeChamber:
    def __init__(self, drift_sign, boundaries):
        self.drift_sign = drift_sign
        self.drift_axis = 0
        self.cathode_pos = 0.0
        self.anode_pos = boundaries[0]
        self.boundaries = np.asarray([boundaries, [-1.0, 1.0], [-1.0, 1.0]])


class FakeModule:
    drift_axis = 0
    cathode_pos = 0.0

    def __init__(self):
        self.chambers = [FakeChamber(1, (-10.0, 0.0)), FakeChamber(-1, (0.0, 10.0))]

    def __getitem__(self, idx):
        return self.chambers[idx]


class FakeTPC:
    def __init__(self):
        self.modules = [FakeModule()]

    def __getitem__(self, idx):
        return self.modules[idx]


class FakeGeo:
    def __init__(self):
        self.tpc = FakeTPC()

    def get_contributors(self, sources):
        sources = np.asarray(sources, dtype=np.int64)
        pairs = np.unique(sources, axis=0)
        return pairs[:, 0], pairs[:, 1]

    def get_volume_index(self, sources, module, tpc):
        sources = np.asarray(sources, dtype=np.int64)
        return np.where((sources[:, 0] == module) & (sources[:, 1] == tpc))[0]

    def get_min_volume_offset(self, points, module, tpc):
        return np.zeros(3, dtype=np.float32)


def make_particle(
    id=0,
    interaction_id=0,
    sources=((0, 0), (0, 0), (0, 1), (0, 1)),
    points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    shape=TRACK_SHP,
    is_truth=False,
):
    points = np.asarray(points, dtype=np.float32)
    particle = SimpleNamespace(
        id=id,
        interaction_id=interaction_id,
        is_truth=is_truth,
        units="cm",
        shape=shape,
        index=np.arange(len(points), dtype=np.int64),
        points=points.copy(),
        sources=np.asarray(sources, dtype=np.int64),
        start_point=points[0].copy(),
        end_point=points[-1].copy(),
        start_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        end_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        is_cathode_crosser=False,
        cathode_offset=0.0,
    )

    def reset_cathode_crosser():
        particle.is_cathode_crosser = False
        particle.cathode_offset = 0.0

    def reset_match():
        particle.is_matched = False

    def merge(other):
        particle.points = np.vstack([particle.points, other.points])
        particle.sources = np.vstack([particle.sources, other.sources])
        particle.index = np.arange(len(particle.points), dtype=np.int64)

    particle.reset_cathode_crosser = reset_cathode_crosser
    particle.reset_match = reset_match
    particle.merge = merge
    return particle


def make_interaction(id=0):
    inter = SimpleNamespace(
        id=id,
        is_cathode_crosser=False,
        cathode_offset=0.0,
        points=np.empty((0, 3), dtype=np.float32),
    )

    def reset_cathode_crosser():
        inter.is_cathode_crosser = False
        inter.cathode_offset = 0.0

    def reset_match():
        inter.is_matched = False

    inter.reset_cathode_crosser = reset_cathode_crosser
    inter.reset_match = reset_match
    return inter


@pytest.fixture(autouse=True)
def patch_geo(monkeypatch):
    monkeypatch.setattr(cathode_mod.GeoManager, "get_instance", lambda: FakeGeo())


def test_cathode_crosser_processor_validates_adjust_method():
    with pytest.raises(AssertionError, match="adjust cathode"):
        cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, adjust_method="bad")


def test_cathode_crosser_processor_tags_offsets_and_adjusts_positions():
    processor = cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, merge_crossers=False)
    particle = make_particle()
    interaction = make_interaction()
    data = {
        "points": particle.points.copy(),
        "reco_particles": [particle],
        "reco_interactions": [interaction],
    }

    result = processor.process(data)

    assert result["reco_particles"] == [particle]
    assert particle.is_cathode_crosser is True
    assert particle.cathode_offset == pytest.approx(1.0)
    np.testing.assert_allclose(particle.points[:, 0], [-1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(data["points"][:, 0], [-1.0, 0.0, 0.0, 1.0])
    assert interaction.is_cathode_crosser is True
    assert interaction.cathode_offset == pytest.approx(1.0)
    np.testing.assert_allclose(interaction.points[:, 0], [-1.0, 0.0, 0.0, 1.0])


def test_cathode_crosser_processor_process_branches(monkeypatch):
    both = cathode_mod.CathodeCrosserProcessor(
        1.0, 1.0, 1.0, merge_crossers=False, run_mode="both"
    )
    assert "points_label" in both.keys

    processor = cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, merge_crossers=False)
    non_track = make_particle(shape=999)
    single_tpc = make_particle(
        sources=((0, 0), (0, 0)), points=((-1.0, 0, 0), (-0.5, 0, 0))
    )
    data = {
        "points": np.vstack([non_track.points, single_tpc.points]),
        "reco_particles": [non_track, single_tpc],
        "reco_interactions": [make_interaction()],
    }

    processor.process(data)

    assert non_track.is_cathode_crosser is False
    assert single_tpc.is_cathode_crosser is False

    merge_processor = cathode_mod.CathodeCrosserProcessor(
        1.0, 1.0, 1.0, merge_crossers=True, adjust_crossers=False
    )
    merged = make_particle(
        sources=((0, 0), (0, 0)), points=((-1.0, 0, 0), (-0.5, 0, 0))
    )
    monkeypatch.setattr(
        merge_processor,
        "find_matches",
        lambda particles: (particles, [make_interaction()]),
    )

    merge_processor.process(
        {
            "points": merged.points.copy(),
            "reco_particles": [merged],
            "reco_interactions": [make_interaction()],
        }
    )

    assert merged.is_matched is False

    bad_direction = make_particle(
        sources=((0, 0), (0, 0)), points=((-1.0, 0, 0), (-0.5, 0, 0))
    )
    bad_direction.start_dir[0] = -np.inf
    with pytest.raises(AssertionError, match="direction"):
        merge_processor.process(
            {
                "points": bad_direction.points.copy(),
                "reco_particles": [bad_direction],
                "reco_interactions": [make_interaction()],
            }
        )


def test_cathode_crosser_processor_skips_empty_particles_and_requires_sources():
    processor = cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, merge_crossers=False)
    interactions = [make_interaction()]
    result = processor.process(
        {
            "points": np.empty((0, 3)),
            "reco_particles": [],
            "reco_interactions": interactions,
        }
    )
    assert result["reco_particles"] == []
    assert result["reco_interactions"] == interactions

    particle = make_particle(sources=np.empty((0, 2), dtype=np.int64))
    with pytest.raises(AssertionError, match="sources"):
        processor.process(
            {
                "points": particle.points.copy(),
                "reco_particles": [particle],
                "reco_interactions": [make_interaction()],
            }
        )


def test_cathode_crosser_processor_offset_helpers(monkeypatch):
    points = np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 4.0]], dtype=np.float32)
    dirs = np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]], dtype=np.float32)
    assert np.isfinite(
        cathode_mod.CathodeCrosserProcessor.projection_offset(points, dirs)
    )
    assert np.isfinite(cathode_mod.CathodeCrosserProcessor.dot_offset(points, dirs))

    with pytest.raises(AssertionError, match="projection method"):
        cathode_mod.CathodeCrosserProcessor.projection_offset(
            points, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )

    with pytest.raises(AssertionError, match="projection method"):
        cathode_mod.CathodeCrosserProcessor.dot_offset(
            points, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )

    monkeypatch.setattr(
        cathode_mod,
        "cluster_direction",
        lambda points, start: np.array([-1.0, 0.1, 0.0], dtype=np.float32),
    )
    processor = cathode_mod.CathodeCrosserProcessor(
        1.0, 1.0, 1.0, merge_crossers=False, adjust_method="projection"
    )
    assert np.isfinite(processor.get_cathode_offset(make_particle()))

    processor = cathode_mod.CathodeCrosserProcessor(
        1.0, 1.0, 1.0, merge_crossers=False, adjust_method="dot"
    )
    assert np.isfinite(processor.get_cathode_offset(make_particle()))


def test_cathode_crosser_processor_find_matches_skip_paths(monkeypatch):
    class FakeRecoInteraction:
        @classmethod
        def from_particles(cls, particles):
            return SimpleNamespace(
                particles=particles,
                particle_ids=np.array([p.id for p in particles], dtype=np.int64),
            )

    monkeypatch.setattr(cathode_mod, "RecoInteraction", FakeRecoInteraction)
    processor = cathode_mod.CathodeCrosserProcessor(
        crossing_point_tolerance=0.1,
        offset_tolerance=0.1,
        angle_tolerance=0.1,
        merge_crossers=True,
        adjust_crossers=False,
    )
    multi_tpc = make_particle(id=0)
    same_tpc = make_particle(
        id=1,
        sources=((0, 0), (0, 0)),
        points=((-1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
    )
    same_tpc_other = make_particle(
        id=2,
        sources=((0, 0), (0, 0)),
        points=((-0.2, 0.0, 0.0), (-0.1, 0.0, 0.0)),
    )
    far_tpc = make_particle(
        id=3,
        sources=((0, 1), (0, 1)),
        points=((10.0, 10.0, 0.0), (11.0, 10.0, 0.0)),
    )

    particles, interactions = processor.find_matches(
        [multi_tpc, same_tpc, same_tpc_other, far_tpc]
    )

    assert len(particles) == 4
    assert all(not part.is_cathode_crosser for part in particles)
    assert len(interactions) == 1


def test_cathode_crosser_processor_merges_particles(monkeypatch):
    class FakeRecoInteraction:
        @classmethod
        def from_particles(cls, particles):
            return SimpleNamespace(
                particles=particles,
                particle_ids=np.array([p.id for p in particles], dtype=np.int64),
            )

    monkeypatch.setattr(cathode_mod, "RecoInteraction", FakeRecoInteraction)
    processor = cathode_mod.CathodeCrosserProcessor(
        crossing_point_tolerance=1.0,
        offset_tolerance=3.0,
        angle_tolerance=np.pi,
        merge_crossers=True,
        adjust_crossers=False,
    )
    left = make_particle(
        id=0,
        interaction_id=0,
        sources=((0, 0), (0, 0)),
        points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    right = make_particle(
        id=1,
        interaction_id=1,
        sources=((0, 1), (0, 1)),
        points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    right.start_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    particles, interactions = processor.find_matches([left, right])

    assert len(particles) == 1
    assert particles[0].is_cathode_crosser is True
    assert particles[0].interaction_id == 0
    assert len(interactions) == 1

    left = make_particle(
        id=0,
        interaction_id=0,
        sources=((0, 0), (0, 0)),
        points=((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    right = make_particle(
        id=1,
        interaction_id=1,
        sources=((0, 1), (0, 1)),
        points=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    right.start_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    particles, _ = processor.find_matches([left, right, make_particle(id=2)])
    assert len(particles) == 2


def test_cathode_crosser_processor_merge_particles_directly():
    processor = cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, merge_crossers=False)
    first = make_particle(
        id=0,
        interaction_id=0,
        sources=((0, 0), (0, 0)),
        points=((-1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
    )
    second = make_particle(
        id=1,
        interaction_id=1,
        sources=((0, 1), (0, 1)),
        points=((0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    sister = make_particle(
        id=2,
        interaction_id=1,
        sources=((0, 1), (0, 1)),
        points=((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )

    particles = processor.merge_particles([first, second, sister], 0, 1)

    assert len(particles) == 2
    assert particles[0].is_cathode_crosser is True
    assert [part.id for part in particles] == [0, 1]
    assert particles[1].interaction_id == 0


def test_cathode_crosser_processor_adjusts_sister_particles():
    processor = cathode_mod.CathodeCrosserProcessor(1.0, 1.0, 1.0, merge_crossers=False)
    crosser = make_particle(id=0, interaction_id=0)
    sister = make_particle(
        id=1,
        interaction_id=0,
        sources=((0, 0), (0, 0)),
        points=((-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
    )
    interaction = make_interaction()
    data = {
        "points": np.vstack([crosser.points, sister.points]),
        "reco_particles": [crosser, sister],
        "reco_interactions": [interaction],
    }
    sister.index = np.array([4, 5], dtype=np.int64)

    processor.adjust_positions(data, [crosser, sister], [interaction], 0, 1.0)

    np.testing.assert_allclose(sister.points[:, 0], [-2.0, -1.0])
    np.testing.assert_allclose(sister.start_point[0], -2.0)
    np.testing.assert_allclose(sister.end_point[0], -1.0)
