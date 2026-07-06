from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spine.post.optical import flash_matching as flash_matching_mod
from spine.post.optical.flash_matching import FlashMatchProcessor


class FakeTPC:
    num_chambers_per_module = 2


class FakeGeo:
    name = "demo"
    tpc = FakeTPC()

    def get_volume_index(self, sources, *volume_ids):
        sources = np.asarray(sources)
        if len(volume_ids) == 1:
            return np.where(sources[:, 0] == volume_ids[0])[0]
        return np.where(
            (sources[:, 0] == volume_ids[0]) & (sources[:, 1] == volume_ids[1])
        )[0]

    def translate(self, points, volume_id, ref_volume_id):
        return points + float(ref_volume_id - volume_id)


class FakeBarycenterMatcher:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def get_matches(self, interactions, flashes):
        self.calls.append((interactions, flashes))
        if not interactions or not flashes:
            return []

        if flashes[0].time == 10.0:
            return [
                (
                    interactions[0],
                    flashes[0],
                    SimpleNamespace(hypothesis=[1.0, 2.0], score=0.75),
                )
            ]

        return [(interactions[0], flashes[0], 0.5)]


class FakeLikelihoodMatcher(FakeBarycenterMatcher):
    def __init__(self, detector, parent_path=None, **kwargs):
        super().__init__(detector=detector, parent_path=parent_path, **kwargs)


class FakeMerger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, flashes):
        merged = [
            SimpleNamespace(
                id=0,
                volume_id=flashes[0].volume_id,
                time=10.0,
                total_pe=sum(f.total_pe for f in flashes),
            )
        ]
        return merged, np.asarray([[0, 1]], dtype=object)


def make_interaction(
    id=0,
    sources=((0, 0), (0, 0)),
    time_contained=True,
    cathode_crosser=False,
    cathode_offset=0.0,
):
    inter = SimpleNamespace(
        id=id,
        is_truth=False,
        units="cm",
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        depositions=np.array([1.0, 2.0], dtype=np.float32),
        sources=np.asarray(sources, dtype=np.int64),
        is_time_contained=time_contained,
        is_cathode_crosser=cathode_crosser,
        cathode_offset=cathode_offset,
        is_flash_matched=True,
        flash_total_pe=99.0,
        flash_hypo_pe=99.0,
    )

    def reset_flash_match():
        inter.is_flash_matched = False
        inter.flash_total_pe = 0.0
        inter.flash_hypo_pe = 0.0
        inter.flash_ids = np.empty(0, dtype=np.int32)
        inter.flash_volume_ids = np.empty(0, dtype=np.int32)
        inter.flash_times = np.empty(0, dtype=np.float32)
        inter.flash_scores = np.empty(0, dtype=np.float32)

    inter.reset_flash_match = reset_flash_match
    return inter


def make_flash(id=0, volume_id=0, time=1.0, total_pe=10.0):
    return SimpleNamespace(id=id, volume_id=volume_id, time=time, total_pe=total_pe)


@pytest.fixture(autouse=True)
def patch_flash_matching(monkeypatch):
    monkeypatch.setattr(
        flash_matching_mod.GeoManager, "get_instance", lambda: FakeGeo()
    )
    monkeypatch.setattr(
        flash_matching_mod, "BarycenterFlashMatcher", FakeBarycenterMatcher
    )
    monkeypatch.setattr(
        flash_matching_mod, "LikelihoodFlashMatcher", FakeLikelihoodMatcher
    )
    monkeypatch.setattr(flash_matching_mod, "FlashMerger", FakeMerger)


def test_flash_match_processor_validates_volume():
    with pytest.raises(ValueError, match="volume"):
        FlashMatchProcessor(flash_key="flashes", volume="bad")


def test_flash_match_processor_validates_method_and_initializes_likelihood():
    processor = FlashMatchProcessor(
        flash_key="flashes",
        volume="module",
        method="likelihood",
        parent_path="cfg",
        foo="bar",
    )

    matcher = cast(FakeLikelihoodMatcher, processor.matcher)
    assert matcher.kwargs == {"detector": "demo", "parent_path": "cfg", "foo": "bar"}

    with pytest.raises(ValueError, match="method"):
        FlashMatchProcessor(flash_key="flashes", volume="module", method="closest")


def test_flash_match_processor_skips_empty_interactions():
    processor = FlashMatchProcessor(
        flash_key="flashes", volume="module", method="barycenter"
    )

    assert (
        processor.process({"flashes": [make_flash()], "reco_interactions": []}) is None
    )


def test_flash_match_processor_matches_module_volume_with_scalar_score():
    processor = FlashMatchProcessor(
        flash_key="flashes", volume="module", method="barycenter"
    )
    inter = make_interaction()
    flash = make_flash(id=4, volume_id=0, time=3.0, total_pe=12.0)

    result = processor.process({"flashes": [flash], "reco_interactions": [inter]})

    assert result is None
    assert inter.is_flash_matched is True
    assert inter.flash_total_pe == 12.0
    assert inter.flash_hypo_pe == -1.0
    assert np.array_equal(inter.flash_ids, np.array([4], dtype=np.int32))
    assert np.array_equal(inter.flash_volume_ids, np.array([0], dtype=np.int32))
    assert np.array_equal(inter.flash_times, np.array([3.0], dtype=np.float32))
    assert np.array_equal(inter.flash_scores, np.array([0.5], dtype=np.float32))

    processor.volume = "detector"
    with pytest.raises(ValueError, match="Volume"):
        processor.process({"flashes": [flash], "reco_interactions": [inter]})


def test_flash_match_processor_filters_and_matches_tpc_volume_with_translation():
    processor = FlashMatchProcessor(
        flash_key="flashes",
        volume="tpc",
        ref_volume_id=3,
        method="barycenter",
        time_contained=True,
        max_cathode_offset=2.0,
    )
    matched = make_interaction(sources=((0, 1), (0, 1)))
    skipped_time = make_interaction(id=1, sources=((0, 1),), time_contained=False)
    skipped_cathode = make_interaction(
        id=2, sources=((0, 1),), cathode_crosser=True, cathode_offset=5.0
    )
    no_points = make_interaction(id=3, sources=((1, 0),))
    flash = make_flash(id=1, volume_id=1)

    processor.process(
        {
            "flashes": [flash],
            "reco_interactions": [matched, skipped_time, skipped_cathode, no_points],
        }
    )

    matcher = cast(FakeBarycenterMatcher, processor.matcher)
    interactions_v, _ = matcher.calls[0]
    np.testing.assert_allclose(interactions_v[0].points, matched.points + 2.0)
    assert matched.is_flash_matched is True
    assert skipped_time.is_flash_matched is False
    assert skipped_cathode.is_flash_matched is False
    assert no_points.is_flash_matched is False
    assert "time_containment" in processor._upstream
    assert "cathode_crosser" in processor._upstream


def test_flash_match_processor_expands_merged_flashes_and_object_score():
    processor = FlashMatchProcessor(
        flash_key="flashes",
        volume="module",
        method="barycenter",
        merge={"window": 1.0},
    )
    inter = make_interaction()
    flashes = [
        make_flash(id=0, volume_id=0, time=1.0, total_pe=10.0),
        make_flash(id=1, volume_id=0, time=2.0, total_pe=5.0),
    ]

    processor.process({"flashes": flashes, "reco_interactions": [inter]})

    assert inter.flash_total_pe == 15.0
    assert inter.flash_hypo_pe == pytest.approx(3.0)
    assert np.array_equal(inter.flash_ids, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(inter.flash_volume_ids, np.array([0, 0], dtype=np.int32))
    assert np.array_equal(inter.flash_times, np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(inter.flash_scores, np.array([0.75, 0.75], dtype=np.float32))


def test_flash_match_processor_accumulates_multiple_matches():
    class MultiMatch(FakeBarycenterMatcher):
        def get_matches(self, interactions, flashes):
            return [(interactions[0], flash, 0.5) for flash in flashes]

    processor = FlashMatchProcessor(
        flash_key="flashes", volume="module", method="barycenter"
    )
    processor.matcher = cast(Any, MultiMatch())
    inter = make_interaction()
    flashes = [
        make_flash(id=0, volume_id=0, total_pe=10.0),
        make_flash(id=1, volume_id=0, total_pe=5.0),
    ]

    processor.process({"flashes": flashes, "reco_interactions": [inter]})

    assert inter.flash_total_pe == 15.0
    assert inter.flash_hypo_pe == -2.0


def test_flash_match_processor_returns_updated_flashes_when_requested():
    processor = FlashMatchProcessor(
        flash_key="flashes",
        volume="module",
        method="barycenter",
        merge={"window": 1.0},
        update_flashes=True,
    )
    inter = make_interaction()
    flashes = [
        make_flash(id=0, volume_id=0, total_pe=10.0),
        make_flash(id=1, volume_id=0, total_pe=5.0),
    ]

    result = processor.process({"flashes": flashes, "reco_interactions": [inter]})

    assert result is not None
    assert len(result["flashes"]) == 1
    assert inter.flash_ids.tolist() == [0]
