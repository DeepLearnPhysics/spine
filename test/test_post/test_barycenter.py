from __future__ import annotations

import numpy as np
import pytest

from spine.post.optical.barycenter import BarycenterFlashMatcher


class FakeFlash:
    def __init__(
        self,
        center,
        width,
        total_pe=10.0,
        time=0.0,
        pe_per_ch=(),
        volume_id=0,
    ):
        self.center = np.asarray(center, dtype=np.float32)
        self.width = np.asarray(width, dtype=np.float32)
        self.total_pe = total_pe
        self.time = time
        self.pe_per_ch = np.asarray(pe_per_ch, dtype=np.float32)
        self.volume_id = volume_id


class FakeInteraction:
    def __init__(self, points, depositions=None):
        self.points = np.asarray(points, dtype=np.float32)
        self.depositions = (
            np.ones(len(points), dtype=np.float32)
            if depositions is None
            else np.asarray(depositions, dtype=np.float32)
        )
        self.size = len(points)


def test_barycenter_flash_matcher_validates_reporting_and_quality_parameters():
    with pytest.raises(ValueError, match="not recognized"):
        BarycenterFlashMatcher(report_mode="closest")

    with pytest.raises(ValueError, match="candidate_distance"):
        BarycenterFlashMatcher(report_mode="all")

    with pytest.raises(ValueError, match="position_errors"):
        BarycenterFlashMatcher(report_mode="best_per_flash", position_errors=0.0)

    with pytest.raises(ValueError, match="chi2_floor"):
        BarycenterFlashMatcher(report_mode="best_per_flash", chi2_floor=0.0)

    with pytest.raises(ValueError, match="max_chi2"):
        BarycenterFlashMatcher(report_mode="best_per_flash", max_chi2=-1.0)

    with pytest.raises(ValueError, match="Optical detector geometry"):
        BarycenterFlashMatcher(report_mode="best_per_flash", angle_error=10.0)

    with pytest.raises(ValueError, match="light_charge_bounds"):
        BarycenterFlashMatcher(
            report_mode="best_per_flash", light_charge_bounds=(2.0, 1.0)
        )

    with pytest.raises(ValueError, match="light_model_cfg"):
        BarycenterFlashMatcher(
            report_mode="best_per_flash", light_charge_bounds=(0.25, 3.0)
        )


def test_barycenter_flash_matcher_finds_best_match():
    matcher = BarycenterFlashMatcher(report_mode="best_per_flash")
    interaction = FakeInteraction([[0.0, 1.0, 1.0], [0.0, 1.2, 1.2]])
    flash = FakeFlash([0.0, 1.1, 1.1], [0.0, 0.1, 0.1])

    matches = matcher.get_matches([interaction], [flash])

    assert matches[0][0] is interaction
    assert matches[0][1] is flash
    result = matches[0][2]
    assert result.distance == pytest.approx(0.0)
    assert result.chi2 == pytest.approx(0.0)
    assert result.score == pytest.approx(1.0e6)
    assert result.hypothesis is None


def test_barycenter_flash_matcher_finds_best_flash_per_interaction():
    matcher = BarycenterFlashMatcher(report_mode="best_per_interaction")
    interactions = [
        FakeInteraction([[0.0, 0.0, 0.0]]),
        FakeInteraction([[0.0, 2.0, 0.0]]),
    ]
    flashes = [
        FakeFlash([0.0, 1.0, 0.0], [0.0, 0.1, 0.1]),
        FakeFlash([0.0, 100.0, 0.0], [0.0, 0.1, 0.1]),
    ]

    matches = matcher.get_matches(interactions, flashes)

    assert len(matches) == 2
    assert matches[0][:2] == (interactions[0], flashes[0])
    assert matches[1][:2] == (interactions[1], flashes[0])


def test_barycenter_flash_matcher_rejects_best_match_above_distance():
    matcher = BarycenterFlashMatcher(
        report_mode="best_per_flash", candidate_distance=0.1
    )
    interaction = FakeInteraction([[0.0, 1.0, 1.0]])
    flash = FakeFlash([0.0, 10.0, 10.0], [0.0, 0.1, 0.1])

    assert matcher.get_matches([interaction], [flash]) == []


def test_barycenter_flash_matcher_filters_inputs():
    matcher = BarycenterFlashMatcher(
        report_mode="best_per_flash", time_window=(0.0, 1.0)
    )
    interaction = FakeInteraction([[0.0, 1.0, 1.0]])
    flash = FakeFlash([0.0, 1.0, 1.0], [0.0, 0.1, 0.1], time=2.0)

    assert matcher.get_matches([interaction], [flash]) == []

    matcher = BarycenterFlashMatcher(report_mode="best_per_flash", min_flash_pe=20.0)
    assert matcher.get_matches([interaction], [flash]) == []

    matcher = BarycenterFlashMatcher(report_mode="best_per_flash", min_inter_size=2)
    assert matcher.get_matches([interaction], [flash]) == []


def test_barycenter_flash_matcher_reports_all_and_uses_charge_weighting():
    matcher = BarycenterFlashMatcher(
        report_mode="all",
        candidate_distance=0.1,
        charge_weighted=True,
        first_flash_only=True,
    )
    interaction = FakeInteraction(
        [[0.0, 1.0, 1.0], [0.0, 3.0, 3.0]], depositions=[1.0, 3.0]
    )
    flash = FakeFlash([0.0, 2.5, 2.5], [0.0, 0.1, 0.1])
    ignored_flash = FakeFlash([0.0, 99.0, 99.0], [0.0, 0.1, 0.1])

    matches = matcher.get_matches([interaction], [flash, ignored_flash])

    assert len(matches) == 1
    assert matches[0][0] is interaction
    assert matches[0][1] is flash
    assert matches[0][2].distance == pytest.approx(0.0)


def test_barycenter_flash_matcher_computes_weighted_spatial_chi2():
    matcher = BarycenterFlashMatcher(
        report_mode="all",
        candidate_distance=2.0,
        position_errors=(1.0, 2.0, 4.0),
    )
    interaction = FakeInteraction(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], depositions=[1.0, 1.0]
    )
    flash = FakeFlash([0.0, 1.0, 0.0], [0.0, 0.1, 0.1])

    result = matcher.get_matches([interaction], [flash])[0][2]

    assert result.distance == pytest.approx(1.0)
    assert result.chi2 == pytest.approx(0.25)
    assert result.score == pytest.approx(4.0)
    np.testing.assert_allclose(result.charge_center, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.charge_width, [0.0, 1.0, 0.0])


def test_barycenter_flash_matcher_applies_optional_chi2_cut():
    interaction = FakeInteraction([[0.0, 0.0, 0.0]])
    flash = FakeFlash([0.0, 1.0, 0.0], [0.0, 0.1, 0.1])

    accepted = BarycenterFlashMatcher(
        report_mode="all", candidate_distance=2.0, max_chi2=1.0
    )
    rejected = BarycenterFlashMatcher(
        report_mode="all", candidate_distance=2.0, max_chi2=0.5
    )

    assert len(accepted.get_matches([interaction], [flash])) == 1
    assert rejected.get_matches([interaction], [flash]) == []


def test_barycenter_flash_matcher_applies_optional_light_charge_cut(
    monkeypatch,
):
    class FakeLightModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []

        def get_hypothesis(self, points, weights=None):
            self.calls.append((np.asarray(points), weights))
            return np.asarray([0.2, 0.3])

    monkeypatch.setattr(
        "spine.post.optical.barycenter.OpT0FinderLightModel", FakeLightModel
    )
    matcher = BarycenterFlashMatcher(
        report_mode="all",
        candidate_distance=1.0,
        light_charge_bounds=(0.5, 1.5),
        light_model_cfg="minimal.cfg",
        charge_scale=2.0,
        detector="demo",
    )
    interaction = FakeInteraction(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], depositions=[1.0, 1.0]
    )
    accepted = FakeFlash([0.0, 0.0, 0.0], [0.0, 0.1, 0.1], total_pe=2.0)
    rejected = FakeFlash([0.0, 0.0, 0.0], [0.0, 0.1, 0.1], total_pe=8.0)

    matches = matcher.get_matches([interaction], [accepted, rejected])

    assert len(matches) == 1
    assert matches[0][1] is accepted
    assert matches[0][2].light_charge_ratio == pytest.approx(1.0)
    np.testing.assert_allclose(matches[0][2].hypothesis, [0.8, 1.2])
    assert matcher.light_model.kwargs["algorithm"] == "SemiAnalyticalModel"
    assert len(matcher.light_model.calls) == 1
    np.testing.assert_allclose(matcher.light_model.calls[0][0], [0.0, 0.0, 0.0])
    assert matcher.light_model.calls[0][1] is None


def test_barycenter_flash_matcher_can_propagate_all_charge_points(monkeypatch):
    class FakeLightModel:
        def __init__(self, **kwargs):
            self.calls = []

        def get_hypothesis(self, points, weights=None):
            self.calls.append((np.asarray(points), np.asarray(weights)))
            return np.asarray([0.2, 0.3])

    monkeypatch.setattr(
        "spine.post.optical.barycenter.OpT0FinderLightModel", FakeLightModel
    )
    matcher = BarycenterFlashMatcher(
        report_mode="all",
        candidate_distance=1.0,
        light_charge_bounds=(0.5, 1.5),
        light_model_cfg="minimal.cfg",
        light_model_use_points=True,
        charge_scale=2.0,
    )
    interaction = FakeInteraction(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [np.nan, 2.0, 0.0]],
        depositions=[1.0, 3.0, 2.0],
    )
    flash = FakeFlash([0.0, 0.5, 0.0], [0.0, 0.1, 0.1], total_pe=2.0)

    matches = matcher.get_matches([interaction], [flash])

    assert len(matches) == 1
    np.testing.assert_allclose(matches[0][2].hypothesis, [1.6, 2.4])
    points, weights = matcher.light_model.calls[0]
    np.testing.assert_allclose(points, [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(weights, [1.0, 3.0])


def test_barycenter_flash_matcher_adds_pca_angle_to_chi2():
    class FakeOpticalVolume:
        positions = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        det_ids = None

    class FakeOptical:
        global_index = False
        num_volumes = 1
        volumes = [FakeOpticalVolume()]

    matcher = BarycenterFlashMatcher(
        report_mode="all",
        candidate_distance=1.0,
        angle_error=30.0,
        optical=FakeOptical(),
    )
    interaction = FakeInteraction([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    flash = FakeFlash(
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.1],
        pe_per_ch=[0.0, 0.0, 1.0, 1.0],
    )

    result = matcher.get_matches([interaction], [flash])[0][2]

    assert result.angle == pytest.approx(90.0)
    assert result.chi2 == pytest.approx(9.0)
    assert result.score == pytest.approx(1.0 / 9.0)


def test_barycenter_flash_matcher_rejects_ambiguous_pca_axis():
    points = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    weights = np.ones(4, dtype=np.float64)

    assert BarycenterFlashMatcher._principal_axis(points, weights) is None
