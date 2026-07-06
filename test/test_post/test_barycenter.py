from __future__ import annotations

import numpy as np
import pytest

from spine.post.optical.barycenter import BarycenterFlashMatcher


class FakeFlash:
    def __init__(self, center, width, total_pe=10.0, time=0.0):
        self.center = np.asarray(center, dtype=np.float32)
        self.width = np.asarray(width, dtype=np.float32)
        self.total_pe = total_pe
        self.time = time


class FakeInteraction:
    def __init__(self, points, depositions=None):
        self.points = np.asarray(points, dtype=np.float32)
        self.depositions = (
            np.ones(len(points), dtype=np.float32)
            if depositions is None
            else np.asarray(depositions, dtype=np.float32)
        )
        self.size = len(points)


def test_barycenter_flash_matcher_validates_threshold_distance():
    with pytest.raises(ValueError, match="not recognized"):
        BarycenterFlashMatcher(match_method="closest")

    with pytest.raises(ValueError, match="match_distance"):
        BarycenterFlashMatcher(match_method="threshold")


def test_barycenter_flash_matcher_finds_best_match():
    matcher = BarycenterFlashMatcher(match_method="best")
    interaction = FakeInteraction([[0.0, 1.0, 1.0], [0.0, 1.2, 1.2]])
    flash = FakeFlash([0.0, 1.1, 1.1], [0.0, 0.1, 0.1])

    matches = matcher.get_matches([interaction], [flash])

    assert matches[0][0] is interaction
    assert matches[0][1] is flash
    assert matches[0][2] == pytest.approx(0.0)


def test_barycenter_flash_matcher_rejects_best_match_above_distance():
    matcher = BarycenterFlashMatcher(match_method="best", match_distance=0.1)
    interaction = FakeInteraction([[0.0, 1.0, 1.0]])
    flash = FakeFlash([0.0, 10.0, 10.0], [0.0, 0.1, 0.1])

    assert matcher.get_matches([interaction], [flash]) == []


def test_barycenter_flash_matcher_filters_inputs():
    matcher = BarycenterFlashMatcher(match_method="best", time_window=(0.0, 1.0))
    interaction = FakeInteraction([[0.0, 1.0, 1.0]])
    flash = FakeFlash([0.0, 1.0, 1.0], [0.0, 0.1, 0.1], time=2.0)

    assert matcher.get_matches([interaction], [flash]) == []

    matcher = BarycenterFlashMatcher(match_method="best", min_flash_pe=20.0)
    assert matcher.get_matches([interaction], [flash]) == []

    matcher = BarycenterFlashMatcher(match_method="best", min_inter_size=2)
    assert matcher.get_matches([interaction], [flash]) == []


def test_barycenter_flash_matcher_threshold_and_charge_weighting():
    matcher = BarycenterFlashMatcher(
        match_method="threshold",
        match_distance=0.1,
        charge_weighted=True,
        first_flash_only=True,
    )
    interaction = FakeInteraction(
        [[0.0, 1.0, 1.0], [0.0, 3.0, 3.0]], depositions=[1.0, 3.0]
    )
    flash = FakeFlash([0.0, 2.5, 2.5], [0.0, 0.1, 0.1])
    ignored_flash = FakeFlash([0.0, 99.0, 99.0], [0.0, 0.1, 0.1])

    matches = matcher.get_matches([interaction], [flash, ignored_flash])

    assert matches == [(interaction, flash, pytest.approx(0.0))]
