"""Tests for derived optical output objects."""

import numpy as np

from spine.data import FlashHypothesis


def test_flash_hypothesis_properties():
    """Hypotheses expose derived PE totals and explicit match state."""
    hypothesis = FlashHypothesis(
        id=2,
        interaction_id=4,
        volume_id=1,
        pe_per_ch=np.array([1.5, 2.5], dtype=np.float32),
    )

    assert hypothesis.total_pe == 4.0
    assert hypothesis.is_matched is False

    hypothesis.flash_ids = np.array([3, 5], dtype=np.int32)
    assert hypothesis.is_matched is True
