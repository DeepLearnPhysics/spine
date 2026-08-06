"""Tests for particle truth post-processing utilities."""

import numpy as np

from spine.constants import INVAL_ID, INVAL_IDX, LOWES_SHP
from spine.utils.particles import get_group_primary_ids, get_interaction_ids


class DummyParticle:
    """Minimal LArCV particle interface for primary-label tests."""

    def __init__(self, group_id, shape, time):
        self._group_id = group_id
        self._shape = shape
        self._time = time

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


def test_interaction_ids_normalize_larcv_sentinels():
    """Invalid LArCV interaction IDs should map to the local -1 sentinel."""

    class InteractionParticle:
        def __init__(self, interaction_id):
            self._interaction_id = interaction_id

        def interaction_id(self):
            return self._interaction_id

    interaction_ids = get_interaction_ids(
        [
            InteractionParticle(0),
            InteractionParticle(INVAL_IDX),
            InteractionParticle(INVAL_ID),
        ]
    )

    np.testing.assert_array_equal(interaction_ids, [0, -1, -1])
