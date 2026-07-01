from __future__ import annotations

import numpy as np
import pytest

from spine.ana.metric.optical import FlashMatchingAna


class FakeInteraction:
    def __init__(
        self,
        *,
        is_truth: bool,
        id: int,
        t: float = 0.0,
        nu_id: int = 1,
        is_flash_matched: bool = True,
    ):
        self.is_truth = is_truth
        self.id = id
        self.size = 3
        self.is_contained = True
        self.topology = "1mu"
        self.nu_id = nu_id
        self.t = t
        self.energy_init = 10.0
        self.is_flash_matched = is_flash_matched
        self.flash_ids = np.array([4], dtype=np.int32)
        self.flash_times = np.array([1.0], dtype=np.float32)
        self.flash_scores = np.array([0.5], dtype=np.float32)
        self.flash_total_pe = 20.0
        self.flash_hypo_pe = 18.0

    def scalar_dict(self, attrs, lengths=None):
        return {attr: getattr(self, attr) for attr in attrs}


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(FlashMatchingAna, "initialize_writer", lambda self, name: None)


def test_flash_matching_ana_validates_configuration():
    with pytest.raises(ValueError, match="Invalid matching mode"):
        FlashMatchingAna(match_mode="bad")

    with pytest.raises(ValueError, match="two values"):
        FlashMatchingAna(time_window=(0.0, 1.0, 2.0))


def test_flash_matching_ana_process_writes_both_directions(monkeypatch):
    rows = []
    monkeypatch.setattr(
        FlashMatchingAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    reco = FakeInteraction(is_truth=False, id=1)
    truth = FakeInteraction(is_truth=True, id=2, t=5.0)
    ana = FlashMatchingAna(time_window=(0.0, 10.0), max_num_flashes=1)

    ana.process(
        {
            "interaction_matches_r2t": [(reco, truth)],
            "interaction_matches_r2t_overlap": [0.75],
            "interaction_matches_t2r": [(truth, reco)],
            "interaction_matches_t2r_overlap": [0.5],
        }
    )

    assert [name for name, _ in rows] == ["reco", "truth"]
    assert rows[0][1]["reco_id"] == 1
    assert rows[0][1]["truth_id"] == 2
    assert rows[0][1]["match_overlap"] == 0.75
    assert rows[1][1]["match_overlap"] == 0.5


def test_flash_matching_ana_filters_unmatched_or_out_of_time(monkeypatch):
    rows = []
    monkeypatch.setattr(
        FlashMatchingAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    reco = FakeInteraction(is_truth=False, id=1, is_flash_matched=False)
    truth = FakeInteraction(is_truth=True, id=2, t=20.0)
    ana = FlashMatchingAna(time_window=(0.0, 10.0))

    ana.process(
        {
            "interaction_matches_r2t": [(reco, truth)],
            "interaction_matches_r2t_overlap": [0.75],
            "interaction_matches_t2r": [(truth, reco)],
            "interaction_matches_t2r_overlap": [0.5],
        }
    )

    assert rows == []


def test_flash_matching_ana_filters_non_neutrino_truth(monkeypatch):
    rows = []
    monkeypatch.setattr(
        FlashMatchingAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    reco = FakeInteraction(is_truth=False, id=1)
    truth = FakeInteraction(is_truth=True, id=2, nu_id=-1)
    ana = FlashMatchingAna()

    ana.process(
        {
            "interaction_matches_r2t": [],
            "interaction_matches_r2t_overlap": [],
            "interaction_matches_t2r": [(truth, reco)],
            "interaction_matches_t2r_overlap": [0.5],
        }
    )

    assert rows == []


def test_flash_matching_ana_uses_default_object_for_missing_match(monkeypatch):
    rows = []
    monkeypatch.setattr(
        FlashMatchingAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    reco = FakeInteraction(is_truth=False, id=1)
    ana = FlashMatchingAna(match_mode="reco_to_truth")

    ana.process(
        {
            "interaction_matches_r2t": [(reco, None)],
            "interaction_matches_r2t_overlap": [0.1],
            "interaction_matches_t2r": [],
            "interaction_matches_t2r_overlap": [],
        }
    )

    assert rows[0][0] == "reco"
    assert rows[0][1]["truth_id"] == -1
    assert rows[0][1]["match_overlap"] == 0.1
