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
