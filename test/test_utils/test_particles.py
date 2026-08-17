"""Tests for particle truth post-processing utilities."""

import numpy as np

from spine.constants import INVAL_ID, INVAL_IDX, LOWES_SHP
from spine.utils.particles import (
    get_group_primary_ids,
    get_interaction_ids,
    get_invalid_index,
)


class DummyParticle:
    """Minimal LArCV particle interface for primary-label tests."""

    def __init__(self, group_id, shape, time, num_voxels=1):
        self._group_id = group_id
        self._shape = shape
        self._time = time
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

    def num_voxels(self):
        return self._num_voxels


def test_group_primary_ids_respect_label_le():
    """Low-energy fragments should be primary only when labels are retained."""
    particles = [
        DummyParticle(group_id=0, shape=LOWES_SHP, time=0.0),
        DummyParticle(group_id=0, shape=0, time=1.0),
    ]
    valid_mask = np.ones(len(particles), dtype=bool)

    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=False), [0, 1]
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
        get_group_primary_ids(particles, valid_mask, label_le=False), [1, 0]
    )
    np.testing.assert_array_equal(
        get_group_primary_ids(particles, valid_mask, label_le=True), [1, 0]
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
