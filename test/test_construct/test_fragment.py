"""Tests for fragment construction."""

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.construct.fragment import FragmentBuilder
from spine.data.larcv.particle import Particle
from spine.data.out import RecoFragment, TruthFragment

from .conftest import make_label_tensor


def test_build_reco_fragments_with_optional_predictions(points, depositions):
    """Reco fragment construction should map cluster and optional predictions."""
    builder = FragmentBuilder(mode="reco", units="px")
    fragments = builder._build_reco(
        points=points,
        depositions=depositions,
        fragment_clusts=[
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
        ],
        fragment_shapes=np.array([0, TRACK_SHP], dtype=np.int32),
        fragment_start_points=np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        fragment_end_points=np.array([[1, 0, 0], [3, 0, 0]], dtype=np.float32),
        fragment_group_pred=np.array([10, 11], dtype=np.int32),
        fragment_node_pred=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        sources=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
        orig_index=np.array([10, 11, 12, 13], dtype=np.int32),
    )

    assert [frag.id for frag in fragments] == [0, 1]
    assert fragments[0].particle_id == 10
    assert fragments[1].particle_id == 11
    assert fragments[0].is_primary is False
    assert fragments[1].is_primary is True
    np.testing.assert_array_equal(fragments[0].sources, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(fragments[1].orig_index, [12, 13])
    assert np.isnan(fragments[0].end_point).all()
    np.testing.assert_array_equal(fragments[1].end_point, [3, 0, 0])


def test_build_truth_fragments_from_labels(points, depositions):
    """Truth fragments should be built from visible non-negative cluster IDs."""
    builder = FragmentBuilder(mode="truth", units="px")
    particle = Particle(id=0, group_id=3, parent_id=4, interaction_id=5)
    labels = make_label_tensor(points, depositions, [0, 0, 1, -1])

    fragments = builder._build_truth(
        particles=[particle, Particle(id=1)],
        label_tensor=labels,
        points_label=points,
        depositions_label=depositions,
    )

    assert len(fragments) == 2
    assert all(isinstance(frag, TruthFragment) for frag in fragments)
    np.testing.assert_array_equal(fragments[0].index, [0, 1])
    np.testing.assert_array_equal(fragments[1].index, [2])
    assert fragments[0].orig_id == 0


def test_build_truth_fragments_reject_invalid_particle_ids(points, depositions):
    """Truth fragment construction should fail on out-of-range particle IDs."""
    builder = FragmentBuilder(mode="truth", units="px")
    labels = make_label_tensor(
        points, depositions, [0, 0, 1, -1], part_ids=[0, 0, 4, -1]
    )

    with pytest.raises(ValueError, match="Invalid particle ID"):
        builder._build_truth(
            particles=[Particle(id=0), Particle(id=1)],
            label_tensor=labels,
            label_adapt_tensor=labels,
            points_label=points,
            depositions_label=depositions,
            points=points,
            depositions=depositions,
        )


def test_build_truth_fragments_truth_only_with_optional_arrays(points, depositions):
    """Truth-only fragments should preserve optional charge/source/orig/G4 data."""
    builder = FragmentBuilder(mode="truth", units="px")
    labels = make_label_tensor(points, depositions, [0, 0, 1, 1])

    fragments = builder.build_truth(
        {
            "label_tensor": labels,
            "points_label": points,
            "depositions_label": depositions,
            "depositions_q_label": depositions + 100,
            "sources_label": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
            "orig_index_label": np.array([10, 11, 12, 13], dtype=np.int32),
            "label_g4_tensor": labels,
            "points_g4": points + 20,
            "depositions_g4": depositions + 20,
        }
    )

    np.testing.assert_array_equal(fragments[0].depositions_q, depositions[[0, 1]] + 100)
    np.testing.assert_array_equal(fragments[0].sources, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(fragments[1].orig_index, [12, 13])
    np.testing.assert_array_equal(fragments[1].points_g4, points[[2, 3]] + 20)


def test_build_truth_fragments_with_adapted_and_g4_labels(points, depositions):
    """Truth fragment construction should fill adapted and G4 long-form arrays."""
    builder = FragmentBuilder(mode="truth", units="px")
    labels = make_label_tensor(points, depositions, [0, 0, 1, 1])
    adapted = make_label_tensor(
        points, depositions, [0, 1, 1, -1], part_ids=[0, 0, 1, -1]
    )
    g4 = make_label_tensor(points + 20, depositions + 20, [0, 1, 1, -1])

    fragments = builder._build_truth(
        particles=[Particle(id=0), Particle(id=1)],
        label_tensor=labels,
        label_adapt_tensor=adapted,
        points_label=points,
        depositions_label=depositions,
        depositions_q_label=depositions + 100,
        points=points + 10,
        depositions=depositions + 10,
        label_g4_tensor=g4,
        points_g4=points + 20,
        depositions_g4=depositions + 20,
        sources=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
    )

    assert len(fragments) == 2
    np.testing.assert_array_equal(fragments[0].index_adapt, [0])
    np.testing.assert_array_equal(fragments[1].points_adapt, points[[1, 2]] + 10)
    np.testing.assert_array_equal(fragments[1].sources_adapt, [[0, 1], [1, 0]])


def test_build_truth_fragments_require_dependent_optional_arrays(points, depositions):
    """Adapted and G4 truth fragment paths should validate their dependencies."""
    builder = FragmentBuilder(mode="truth", units="px")
    labels = make_label_tensor(points, depositions, [0, 0, 1, 1])

    with pytest.raises(ValueError, match="adapted truth fragments"):
        builder._build_truth(
            label_tensor=labels,
            label_adapt_tensor=labels,
            points_label=points,
            depositions_label=depositions,
        )

    with pytest.raises(ValueError, match="Geant4 points and depositions"):
        builder._build_truth(
            label_tensor=labels,
            points_label=points,
            depositions_label=depositions,
            label_g4_tensor=labels,
            points_g4=points + 20,
        )


def test_load_reco_fragment_restores_long_form_arrays(points, depositions):
    """Loaded reco fragments should recover skipped long-form arrays."""
    builder = FragmentBuilder(mode="reco", units="px")
    fragment = RecoFragment(id=0, index=np.array([1, 3], dtype=np.int32))

    sources = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)

    result = builder.load_reco(
        {
            "reco_fragments": [fragment],
            "points": points,
            "depositions": depositions,
            "sources": sources,
        }
    )

    np.testing.assert_array_equal(result[0].points, points[[1, 3]])
    np.testing.assert_array_equal(result[0].depositions, depositions[[1, 3]])
    np.testing.assert_array_equal(result[0].sources, sources[[1, 3]])


def test_load_reco_fragment_requires_depositions_when_points_are_provided(points):
    """Reco fragment loading should validate dependent long-form arrays."""
    builder = FragmentBuilder(mode="reco", units="px")
    fragment = RecoFragment(id=0, index=np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="reco fragments if points are provided"):
        builder._load_reco([fragment], points=points)


def test_load_truth_fragment_restores_all_available_long_form_arrays(
    points, depositions
):
    """Loaded truth fragments should recover true, adapted and G4 arrays."""
    builder = FragmentBuilder(mode="truth", units="px")
    fragment = TruthFragment(
        id=0,
        index=np.array([0, 2], dtype=np.int32),
        index_adapt=np.array([1], dtype=np.int32),
        index_g4=np.array([3], dtype=np.int32),
    )

    result = builder._load_truth(
        [fragment],
        points_label=points,
        depositions_label=depositions,
        depositions_q_label=depositions + 100,
        points=points + 10,
        depositions=depositions + 10,
        points_g4=points + 20,
        depositions_g4=depositions + 20,
        sources_label=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
        sources=np.array([[1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.int32),
    )

    np.testing.assert_array_equal(result[0].points, points[[0, 2]])
    np.testing.assert_array_equal(result[0].depositions_q, depositions[[0, 2]] + 100)
    np.testing.assert_array_equal(result[0].points_adapt, points[[1]] + 10)
    np.testing.assert_array_equal(result[0].points_g4, points[[3]] + 20)
    np.testing.assert_array_equal(result[0].sources, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(result[0].sources_adapt, [[1, 1]])


def test_load_fragment_rejects_out_of_order_ids():
    """Stored fragments should preserve sequential IDs when loading."""
    builder = FragmentBuilder(mode="reco", units="px")

    with pytest.raises(ValueError, match="stored fragments"):
        builder._load_reco([RecoFragment(id=2)])

    builder = FragmentBuilder(mode="truth", units="px")
    with pytest.raises(ValueError, match="stored fragments"):
        builder._load_truth([TruthFragment(id=2)])


def test_load_truth_fragment_requires_matching_deposition_arrays(points, depositions):
    """Truth fragment loading should validate dependent long-form arrays."""
    builder = FragmentBuilder(mode="truth", units="px")
    fragment = TruthFragment(
        id=0,
        index=np.array([0], dtype=np.int32),
        index_adapt=np.array([0], dtype=np.int32),
        index_g4=np.array([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="label points are provided"):
        builder._load_truth([fragment], points_label=points)

    with pytest.raises(ValueError, match="adapted truth fragments"):
        builder._load_truth([fragment], points=points)

    with pytest.raises(ValueError, match="Geant4 truth fragments"):
        builder._load_truth([fragment], points_g4=points)
