"""Tests for interaction construction."""

import numpy as np
import pytest

from spine.construct.interaction import InteractionBuilder
from spine.data.larcv.neutrino import Neutrino
from spine.data.out import (
    RecoFragment,
    RecoInteraction,
    RecoParticle,
    TruthInteraction,
    TruthParticle,
)


def test_build_reco_interactions_from_particles():
    """Reco interactions should group particles and renumber interaction IDs."""
    builder = InteractionBuilder(mode="reco", units="px")
    frag = RecoFragment(id=0)
    particles = [
        RecoParticle(id=0, interaction_id=5, index=np.array([0]), fragments=[frag]),
        RecoParticle(id=1, interaction_id=5, index=np.array([1])),
        RecoParticle(id=2, interaction_id=7, index=np.array([2])),
    ]

    interactions = builder._build_reco(particles)

    assert len(interactions) == 2
    assert interactions[0].particle_ids.tolist() == [0, 1]
    assert interactions[1].particle_ids.tolist() == [2]
    assert particles[0].interaction_id == 0
    assert frag.interaction_id == 0


def test_build_reco_interactions_rejects_invalid_ids():
    """Reco interactions require valid non-negative interaction IDs."""
    builder = InteractionBuilder(mode="reco", units="px")

    with pytest.raises(ValueError, match="Invalid reconstructed"):
        builder.build_reco({"reco_particles": [RecoParticle(id=0, interaction_id=-1)]})


def test_build_truth_interactions_without_neutrinos():
    """Truth interactions should use ancestor positions as fallback vertices."""
    builder = InteractionBuilder(mode="truth", units="px")
    frag = RecoFragment(id=0)
    particles = [
        TruthParticle(
            id=0,
            interaction_id=4,
            nu_id=-1,
            index=np.array([0]),
            ancestor_position=np.array([1, 2, 3], dtype=np.float32),
            fragments=[frag],
        ),
        TruthParticle(
            id=1,
            interaction_id=4,
            nu_id=-1,
            index=np.array([1]),
            ancestor_position=np.array([1, 2, 3], dtype=np.float32),
        ),
    ]

    interactions = builder._build_truth(particles)

    assert len(interactions) == 1
    assert interactions[0].orig_id == 4
    assert interactions[0].particle_ids.tolist() == [0, 1]
    np.testing.assert_array_equal(interactions[0].vertex, [1, 2, 3])
    assert particles[0].orig_interaction_id == 4
    assert particles[0].interaction_id == 0
    assert frag.interaction_id == 0


def test_build_truth_interactions_can_attach_neutrino():
    """Truth interactions should copy provided neutrino metadata."""
    builder = InteractionBuilder(mode="truth", units="px")
    particles = [TruthParticle(id=0, interaction_id=0, nu_id=0, index=np.array([0]))]
    neutrino = Neutrino(id=0, position=np.array([9, 8, 7], dtype=np.float32))

    interactions = builder._build_truth(particles, neutrinos=[neutrino])

    np.testing.assert_array_equal(interactions[0].vertex, [9, 8, 7])


def test_build_truth_interactions_reject_invalid_neutrino_ids():
    """Truth interactions should fail on out-of-range neutrino references."""
    builder = InteractionBuilder(mode="truth", units="px")
    particles = [TruthParticle(id=0, interaction_id=0, nu_id=2, index=np.array([0]))]

    with pytest.raises(ValueError, match="Invalid neutrino ID"):
        builder._build_truth(particles, neutrinos=[Neutrino(id=0)])


def test_build_truth_interactions_reject_mixed_neutrino_ids():
    """Truth interactions require a unique neutrino ID per interaction."""
    builder = InteractionBuilder(mode="truth", units="px")
    particles = [
        TruthParticle(id=0, interaction_id=0, nu_id=0),
        TruthParticle(id=1, interaction_id=0, nu_id=1),
    ]

    with pytest.raises(ValueError, match="different neutrino IDs"):
        builder._build_truth(particles)


def test_build_truth_interactions_warns_on_inconsistent_vertices():
    """Truth interactions should warn and still choose a fallback vertex."""
    builder = InteractionBuilder(mode="truth", units="px")
    particles = [
        TruthParticle(
            id=0, interaction_id=0, nu_id=-1, ancestor_position=np.array([1, 2, 3])
        ),
        TruthParticle(
            id=1, interaction_id=0, nu_id=-1, ancestor_position=np.array([2, 2, 3])
        ),
    ]

    with pytest.warns(UserWarning, match="different ancestor positions"):
        interactions = builder.build_truth({"truth_particles": particles})

    np.testing.assert_array_equal(interactions[0].vertex, [2, 2, 3])


def test_load_reco_interaction_restores_particles_and_cat_attrs():
    """Loaded interactions should recover particle references and long arrays."""
    builder = InteractionBuilder(mode="reco", units="px")
    interaction = RecoInteraction(id=0, particle_ids=np.array([0, 1], dtype=np.int32))
    particles = [
        RecoParticle(id=0, index=np.array([0]), orig_index=np.array([10])),
        RecoParticle(id=1, index=np.array([1]), orig_index=np.array([11])),
    ]

    result = builder.load_reco(
        {"reco_interactions": [interaction], "reco_particles": particles}
    )

    assert result[0].particles == particles
    np.testing.assert_array_equal(result[0].index, [0, 1])
    np.testing.assert_array_equal(result[0].orig_index, [10, 11])


def test_load_interaction_rejects_invalid_stored_content():
    """Stored interactions should preserve IDs and contain particles."""
    builder = InteractionBuilder(mode="reco", units="px")
    with pytest.raises(ValueError, match="stored interactions"):
        builder._load_reco([RecoInteraction(id=2)], [RecoParticle(id=0)])

    with pytest.raises(ValueError, match=">= 1 particle"):
        builder._load_reco(
            [RecoInteraction(id=0, particle_ids=np.array([], dtype=np.int32))], []
        )

    builder = InteractionBuilder(mode="truth", units="px")
    with pytest.raises(ValueError, match="stored interactions"):
        builder._load_truth([TruthInteraction(id=2)], [TruthParticle(id=0)])

    with pytest.raises(ValueError, match=">= 1 particle"):
        builder._load_truth(
            [TruthInteraction(id=0, particle_ids=np.array([], dtype=np.int32))], []
        )


def test_load_truth_interaction_restores_particles_and_cat_attrs():
    """Loaded truth interactions should recover particle references and arrays."""
    builder = InteractionBuilder(mode="truth", units="px")
    interaction = TruthInteraction(id=0, particle_ids=np.array([0, 1], dtype=np.int32))
    particles = [
        TruthParticle(id=0, index=np.array([0]), orig_index=np.array([10])),
        TruthParticle(id=1, index=np.array([1]), orig_index=np.array([11])),
    ]

    result = builder.load_truth(
        {"truth_interactions": [interaction], "truth_particles": particles}
    )

    assert result[0].particles == particles
    np.testing.assert_array_equal(result[0].index, [0, 1])
    np.testing.assert_array_equal(result[0].orig_index, [10, 11])
