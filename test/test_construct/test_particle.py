"""Tests for particle construction."""

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.construct.particle import ParticleBuilder
from spine.data.larcv.particle import Particle
from spine.data.out import RecoFragment, RecoParticle, TruthFragment, TruthParticle

from .conftest import make_label_tensor


def test_build_reco_particles_without_orientation_logits(points, depositions):
    """Reco particle construction should not require orientation logits."""
    builder = ParticleBuilder(mode="reco", units="px")
    fragments = [RecoFragment(id=0, particle_id=5), RecoFragment(id=1, particle_id=7)]

    particles = builder._build_reco(
        points=points,
        depositions=depositions,
        particle_clusts=[
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
        ],
        particle_shapes=np.array([0, TRACK_SHP], dtype=np.int32),
        particle_start_points=np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        particle_end_points=np.array([[1, 0, 0], [3, 0, 0]], dtype=np.float32),
        particle_group_pred=np.array([20, 21], dtype=np.int32),
        particle_node_type_pred=np.eye(2, 6, dtype=np.float32),
        particle_node_primary_pred=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        reco_fragments=fragments,
        sources=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
        orig_index=np.array([10, 11, 12, 13], dtype=np.int32),
    )

    assert [part.id for part in particles] == [0, 1]
    assert particles[0].pid == 0
    assert particles[1].pid == 1
    assert particles[0].is_primary is False
    assert particles[1].is_primary is True
    assert particles[0].fragment_ids.tolist() == [0]
    assert particles[1].fragment_ids.tolist() == [1]
    assert fragments[0].particle_id == 0
    assert fragments[1].interaction_id == 21
    np.testing.assert_array_equal(particles[0].sources, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(particles[1].orig_index, [12, 13])


def test_build_reco_track_orientation_can_flip_endpoints(points, depositions):
    """Orientation class 0 should swap track start and end points."""
    builder = ParticleBuilder(mode="reco", units="px")

    particles = builder._build_reco(
        points=points,
        depositions=depositions,
        particle_clusts=[np.array([0, 1], dtype=np.int32)],
        particle_shapes=np.array([TRACK_SHP], dtype=np.int32),
        particle_start_points=np.array([[0, 0, 0]], dtype=np.float32),
        particle_end_points=np.array([[3, 0, 0]], dtype=np.float32),
        particle_group_pred=np.array([0], dtype=np.int32),
        particle_node_type_pred=np.array([[0, 1, 0, 0, 0, 0]], dtype=np.float32),
        particle_node_primary_pred=np.array([[0.0, 1.0]], dtype=np.float32),
        particle_node_orient_pred=np.array([[3.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(particles[0].start_point, [3, 0, 0])
    np.testing.assert_array_equal(particles[0].end_point, [0, 0, 0])


def test_build_truth_particles_from_groups(points, depositions):
    """Truth particle construction should remap visible group IDs."""
    builder = ParticleBuilder(mode="truth", units="px")
    labels = make_label_tensor(
        points, depositions, [0, 0, 1, 1], group_ids=[0, 0, 1, 1]
    )
    particles = [
        Particle(
            id=0,
            group_id=0,
            interaction_id=0,
            parent_id=-1,
            shape=TRACK_SHP,
            pid=2,
            interaction_primary=1,
            distance_travel=12.0,
            first_step=np.array([0, 0, 0], dtype=np.float32),
            last_step=np.array([1, 0, 0], dtype=np.float32),
            energy_deposit=1.5,
        ),
        Particle(
            id=1,
            group_id=1,
            interaction_id=0,
            parent_id=0,
            shape=0,
            pid=1,
            energy_deposit=2.5,
        ),
    ]
    fragments = [
        TruthFragment(id=0, orig_group_id=0),
        TruthFragment(id=1, orig_group_id=1),
    ]

    result = builder.build_truth(
        {
            "particles": particles,
            "label_tensor": labels,
            "points_label": points,
            "depositions_label": depositions,
            "depositions_q_label": depositions + 100,
            "sources_label": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
            "orig_index_label": np.array([10, 11, 12, 13], dtype=np.int32),
            "label_adapt_tensor": labels,
            "points": points + 10,
            "depositions": depositions + 10,
            "sources": np.array([[2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.int32),
            "label_g4_tensor": labels,
            "points_g4": points + 20,
            "depositions_g4": depositions + 20,
            "graph_label": np.array([[0, 1], [0, 99]], dtype=np.int64),
            "truth_fragments": fragments,
        }
    )

    assert len(result) == 2
    assert result[0].orig_id == 0
    assert result[0].children_id.tolist() == [1]
    assert result[1].parent_id == 0
    assert result[0].fragment_ids.tolist() == [0]
    assert fragments[1].particle_id == 1
    np.testing.assert_array_equal(result[0].points_adapt, points[[0, 1]] + 10)
    np.testing.assert_array_equal(result[1].points_g4, points[[2, 3]] + 20)
    np.testing.assert_array_equal(result[1].sources_adapt, [[3, 0], [3, 1]])


def test_build_truth_particles_reject_invalid_group_ids(points, depositions):
    """Truth particle construction should fail on out-of-range group IDs."""
    builder = ParticleBuilder(mode="truth", units="px")
    labels = make_label_tensor(
        points, depositions, [0, 0, 1, 1], group_ids=[0, 0, 5, 5]
    )

    with pytest.raises(ValueError, match="Invalid group ID"):
        builder.build_truth(
            {
                "particles": [Particle(id=0, group_id=0), Particle(id=1, group_id=1)],
                "label_tensor": labels,
                "points_label": points,
                "depositions_label": depositions,
            }
        )


def test_build_truth_particles_reject_wrong_particle_order(points, depositions):
    """Truth particle construction should fail if particle IDs do not align."""
    builder = ParticleBuilder(mode="truth", units="px")
    labels = make_label_tensor(
        points, depositions, [0, 0, -1, -1], group_ids=[0, 0, -1, -1]
    )

    with pytest.raises(ValueError, match="ordering of the true particles"):
        builder.build_truth(
            {
                "particles": [Particle(id=3, group_id=0)],
                "label_tensor": labels,
                "points_label": points,
                "depositions_label": depositions,
            }
        )


def test_build_truth_particles_require_dependent_optional_arrays(points, depositions):
    """Adapted and G4 truth particle paths should validate dependencies."""
    builder = ParticleBuilder(mode="truth", units="px")
    labels = make_label_tensor(
        points, depositions, [0, 0, 1, 1], group_ids=[0, 0, 1, 1]
    )
    particles = [Particle(id=0, group_id=0), Particle(id=1, group_id=1)]

    with pytest.raises(ValueError, match="adapted truth particles"):
        builder.build_truth(
            {
                "particles": particles,
                "label_tensor": labels,
                "label_adapt_tensor": labels,
                "points_label": points,
                "depositions_label": depositions,
            }
        )

    with pytest.raises(ValueError, match="Geant4 points and depositions"):
        builder.build_truth(
            {
                "particles": particles,
                "label_tensor": labels,
                "label_g4_tensor": labels,
                "points_label": points,
                "depositions_label": depositions,
                "points_g4": points + 20,
            }
        )


def test_load_reco_particle_restores_long_form_and_fragments(points, depositions):
    """Loaded reco particles should recover arrays and fragment references."""
    builder = ParticleBuilder(mode="reco", units="px")
    particle = RecoParticle(
        id=0,
        index=np.array([0, 2], dtype=np.int32),
        fragment_ids=np.array([1], dtype=np.int32),
    )
    fragments = [RecoFragment(id=0), RecoFragment(id=1)]

    sources = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    result = builder.load_reco(
        {
            "reco_particles": [particle],
            "points": points,
            "depositions": depositions,
            "sources": sources,
            "reco_fragments": fragments,
        }
    )

    np.testing.assert_array_equal(result[0].points, points[[0, 2]])
    np.testing.assert_array_equal(result[0].sources, sources[[0, 2]])
    assert result[0].fragments == [fragments[1]]


def test_load_truth_particle_restores_all_long_form_arrays(points, depositions):
    """Loaded truth particles should recover true, adapted, G4 and fragments."""
    builder = ParticleBuilder(mode="truth", units="px")
    particle = TruthParticle(
        id=0,
        index=np.array([0, 1], dtype=np.int32),
        index_adapt=np.array([2], dtype=np.int32),
        index_g4=np.array([3], dtype=np.int32),
        fragment_ids=np.array([0], dtype=np.int32),
    )
    fragment = TruthFragment(id=0)

    result = builder.load_truth(
        {
            "truth_particles": [particle],
            "points_label": points,
            "depositions_label": depositions,
            "depositions_q_label": depositions + 100,
            "points": points + 10,
            "depositions": depositions + 10,
            "points_g4": points + 20,
            "depositions_g4": depositions + 20,
            "sources_label": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
            "sources": np.array([[2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.int32),
            "truth_fragments": [fragment],
        }
    )

    np.testing.assert_array_equal(result[0].points, points[[0, 1]])
    np.testing.assert_array_equal(result[0].points_adapt, points[[2]] + 10)
    np.testing.assert_array_equal(result[0].points_g4, points[[3]] + 20)
    assert result[0].fragments == [fragment]


def test_load_particle_rejects_out_of_order_ids():
    """Stored particles should preserve sequential IDs when loading."""
    builder = ParticleBuilder(mode="reco", units="px")

    with pytest.raises(ValueError, match="stored particles"):
        builder._load_reco([RecoParticle(id=2)])

    builder = ParticleBuilder(mode="truth", units="px")
    with pytest.raises(ValueError, match="stored particles"):
        builder._load_truth([TruthParticle(id=2)])


def test_load_particle_requires_matching_deposition_arrays(points):
    """Particle loading should validate dependent long-form arrays."""
    builder = ParticleBuilder(mode="reco", units="px")
    reco = RecoParticle(id=0, index=np.array([0], dtype=np.int32))
    with pytest.raises(ValueError, match="reco particles if points are provided"):
        builder._load_reco([reco], points=points)

    builder = ParticleBuilder(mode="truth", units="px")
    truth = TruthParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        index_adapt=np.array([0], dtype=np.int32),
        index_g4=np.array([0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="label points are provided"):
        builder._load_truth([truth], points_label=points)

    with pytest.raises(ValueError, match="adapted truth particles"):
        builder._load_truth([truth], points=points)

    with pytest.raises(ValueError, match="Geant4 truth particles"):
        builder._load_truth([truth], points_g4=points)
