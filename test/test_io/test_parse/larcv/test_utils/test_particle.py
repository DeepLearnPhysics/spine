"""Tests for particle truth post-processing utilities."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import INVAL_ID, INVAL_IDX, INVAL_TID, LOWES_SHP, MICHL_SHP
from spine.io.parse.larcv.utils.particle import (
    get_group_primary_ids,
    get_inter_primary_ids,
    get_interaction_ids,
    get_invalid_index,
    get_nu_ids,
    get_particle_ids,
    get_valid_mask,
    process_particle_event,
)


class DummyParticle:
    """Minimal LArCV particle interface for primary-label tests."""

    def __init__(
        self,
        group_id,
        shape,
        time,
        *,
        interaction_id=0,
        ancestor_tid=1,
        process="primary",
        pdg=13,
        track_id=1,
        parent_pdg=0,
        parent_tid=0,
        parent_id=0,
        position=(0, 0, 0),
        num_voxels=1,
    ):
        self._group_id = group_id
        self._shape = shape
        self._time = time
        self._interaction_id = interaction_id
        self._ancestor_tid = ancestor_tid
        self._process = process
        self._pdg = pdg
        self._track_id = track_id
        self._parent_pdg = parent_pdg
        self._parent_tid = parent_tid
        self._parent_id = parent_id
        self._position = position
        self._num_voxels = num_voxels

    def group_id(self):
        return self._group_id

    def shape(self):
        return self._shape

    def t(self):
        return self._time

    def first_step(self):
        class Step:
            def __init__(self, time):
                self._time = time

            def t(self):
                return self._time

        return Step(self._time)

    def interaction_id(self):
        return self._interaction_id

    def ancestor_track_id(self):
        return self._ancestor_tid

    def ancestor_creation_process(self):
        return self._process

    def ancestor_position(self):
        x, y, z = self._position
        return SimpleNamespace(x=lambda: x, y=lambda: y, z=lambda: z)

    def parent_pdg_code(self):
        return self._parent_pdg

    def parent_id(self):
        return self._parent_id

    def parent_track_id(self):
        return self._parent_tid

    def track_id(self):
        return self._track_id

    def pdg_code(self):
        return self._pdg

    def position(self):
        return self.ancestor_position()

    def num_voxels(self):
        return self._num_voxels


def test_group_primary_ids_respect_label_le():
    """An excluded low-energy progenitor must not promote its daughter."""
    particles = [
        DummyParticle(group_id=0, shape=LOWES_SHP, time=0.0),
        DummyParticle(group_id=0, shape=0, time=1.0),
    ]
    valid_mask = np.ones(len(particles), dtype=bool)

    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=False), [0, 0]
    )
    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=True), [1, 0]
    )


def test_group_primary_ids_do_not_promote_visible_fragments():
    """An invisible physical primary must not promote a visible fragment."""
    particles = [
        DummyParticle(group_id=0, shape=0, time=-9.2e12, num_voxels=0),
        DummyParticle(group_id=0, shape=0, time=1.0, num_voxels=10),
    ]
    valid_mask = np.ones(len(particles), dtype=bool)

    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=False), [0, 0]
    )
    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=True), [0, 0]
    )


def test_group_primary_ids_require_unique_earliest_progenitor():
    """Only a uniquely earliest visible progenitor is a clean target."""
    valid_mask = np.ones(2, dtype=bool)

    clean = [
        DummyParticle(group_id=0, shape=0, time=0.0),
        DummyParticle(group_id=0, shape=0, time=1.0),
    ]
    np.testing.assert_array_equal(
        get_group_primary_ids(clean, valid_mask, label_le=False), [1, 0]
    )

    earlier_daughter = [
        DummyParticle(group_id=0, shape=0, time=1.0),
        DummyParticle(group_id=0, shape=0, time=0.0),
    ]
    np.testing.assert_array_equal(
        get_group_primary_ids(earlier_daughter, valid_mask, label_le=False), [0, 0]
    )

    tied = [
        DummyParticle(group_id=0, shape=0, time=0.0),
        DummyParticle(group_id=0, shape=0, time=0.0),
    ]
    np.testing.assert_array_equal(
        get_group_primary_ids(tied, valid_mask, label_le=False), [0, 0]
    )


def test_group_primary_ids_use_explicit_visibility():
    """Explicit retained indexes override raw voxel visibility."""
    particles = [
        DummyParticle(group_id=0, shape=0, time=0.0),
        DummyParticle(group_id=0, shape=0, time=1.0),
    ]
    valid_mask = np.ones(2, dtype=bool)

    np.testing.assert_array_equal(
        get_group_primary_ids(
            particles, valid_mask, label_le=False, visible_ids=np.array([1])
        ),
        [0, 0],
    )


def test_interaction_ids_normalize_event_sentinel():
    """Interaction IDs should use the sentinel convention of their event."""

    class InteractionParticle:
        def __init__(self, interaction_id):
            self._interaction_id = interaction_id

        def interaction_id(self):
            return self._interaction_id

    old_ids = get_interaction_ids(
        [InteractionParticle(0), InteractionParticle(INVAL_IDX)]
    )
    new_ids = get_interaction_ids(
        [InteractionParticle(INVAL_IDX), InteractionParticle(INVAL_ID)]
    )

    assert get_invalid_index(np.array([0, INVAL_IDX])) == INVAL_IDX
    assert get_invalid_index(np.array([INVAL_IDX, INVAL_ID])) == INVAL_ID
    np.testing.assert_array_equal(old_ids, [0, -1])
    np.testing.assert_array_equal(new_ids, [INVAL_IDX, -1])


def test_validity_and_ancestor_position_interactions():
    """Validity should fall back to ancestry when interaction IDs are absent."""
    assert get_valid_mask([]).shape == (0,)
    assert get_interaction_ids([]).shape == (0,)
    particles = [
        DummyParticle(
            0,
            0,
            0,
            interaction_id=INVAL_IDX,
            ancestor_tid=1,
            position=(1, 2, 3),
        ),
        DummyParticle(
            1,
            0,
            0,
            interaction_id=INVAL_IDX,
            ancestor_tid=INVAL_TID,
            process="",
            position=(4, 5, 6),
        ),
    ]
    np.testing.assert_array_equal(get_valid_mask(particles), [True, False])
    np.testing.assert_array_equal(get_interaction_ids(particles), [0, -1])


def test_neutrino_ids_from_ids_and_positions():
    """Neutrino association should support explicit IDs and position matching."""
    particles = [DummyParticle(0, 0, 0, position=(1, 2, 3))]
    groups = np.array([0])
    interactions = np.array([4])
    neutrinos = [
        SimpleNamespace(
            interaction_id=lambda: 4,
            position=lambda: particles[0].ancestor_position(),
        )
    ]
    np.testing.assert_array_equal(
        get_nu_ids(particles, groups, interactions, neutrinos=neutrinos), [0]
    )

    position_only = [SimpleNamespace(position=lambda: particles[0].ancestor_position())]
    with pytest.warns(UserWarning, match="floating point"):
        ids = get_nu_ids(particles, groups, interactions, neutrinos=position_only)
    np.testing.assert_array_equal(ids, [0])
    with pytest.raises(AssertionError, match="both"):
        get_nu_ids(particles, groups, interactions, particles, neutrinos)
    assert get_nu_ids([], np.empty(0), np.empty(0)).shape == (0,)


def test_primary_and_particle_id_error_branches():
    """Malformed groups, short-lived parents, and known PDGs should be labeled."""
    bad = [DummyParticle(4, 0, 0)]
    with pytest.warns(UserWarning, match="Bad group ID"):
        np.testing.assert_array_equal(
            get_group_primary_ids(bad, np.ones(1, bool)), [-1]
        )

    invalid = [DummyParticle(INVAL_ID, 0, 0)]
    np.testing.assert_array_equal(
        get_group_primary_ids(invalid, np.ones(1, bool)), [-1]
    )
    michel = [DummyParticle(0, MICHL_SHP, 0)]
    np.testing.assert_array_equal(get_group_primary_ids(michel, np.ones(1, bool)), [1])

    special = [
        DummyParticle(
            0,
            0,
            0,
            parent_pdg=111,
            parent_tid=7,
            ancestor_tid=7,
            pdg=13,
        )
    ]
    np.testing.assert_array_equal(get_inter_primary_ids(special, np.ones(1, bool)), [1])
    assert get_particle_ids(special, np.ones(1, bool))[0] >= 0

    with pytest.warns(UserWarning, match="Bad group ID"):
        np.testing.assert_array_equal(
            get_inter_primary_ids(bad, np.ones(1, bool)), [-1]
        )
    with pytest.warns(UserWarning, match="Bad group ID"):
        np.testing.assert_array_equal(get_particle_ids(bad, np.ones(1, bool)), [-1])
    np.testing.assert_array_equal(get_particle_ids(invalid, np.ones(1, bool)), [-1])


def test_particle_event_sources_invalid_interactions_and_default_masks():
    """Event wrappers, reference sources, and inferred validity masks are covered."""

    class Event:
        def __init__(self, values):
            self.values = values

        def as_vector(self):
            return self.values

    particles = [DummyParticle(0, 0, 0, interaction_id=0, parent_id=INVAL_ID)]
    mpv = [DummyParticle(0, 0, 0, position=(0, 0, 0))]
    with pytest.warns(UserWarning, match="floating point"):
        result = process_particle_event(Event(particles), Event(mpv))
    assert result[0][0] == -1
    process_particle_event(Event(particles), neutrino_event=Event([]))

    # Invalid interactions are skipped for both ID- and position-based matching.
    neutrino = SimpleNamespace(
        interaction_id=lambda: 4,
        position=lambda: particles[0].ancestor_position(),
    )
    assert (
        get_nu_ids(particles, np.array([0]), np.array([-1]), neutrinos=[neutrino])[0]
        == -1
    )
    position_only = SimpleNamespace(position=lambda: particles[0].ancestor_position())
    with pytest.warns(UserWarning, match="floating point"):
        assert (
            get_nu_ids(
                particles, np.array([0]), np.array([-1]), neutrinos=[position_only]
            )[0]
            == -1
        )

    assert get_group_primary_ids(particles).shape == (1,)
    assert get_inter_primary_ids(particles).shape == (1,)
    assert get_particle_ids(particles).shape == (1,)
