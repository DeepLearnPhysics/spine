from __future__ import annotations

import pytest

from spine.ana.script.save import SaveAna


class FakeObject:
    def __init__(self, id: int, value: float):
        self.id = id
        self.value = value

    def scalar_dict(self, attrs=None, lengths=None):
        keys = ("id", "value") if attrs is None else attrs
        return {key: getattr(self, key) for key in keys}


class FakeWriter:
    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fake_writers(monkeypatch):
    def initialize_writer(self, name):
        self.writers[name] = FakeWriter()

    monkeypatch.setattr(SaveAna, "initialize_writer", initialize_writer)


def test_save_ana_validates_configuration():
    with pytest.raises(ValueError, match="Invalid matching mode"):
        SaveAna(obj_type="particle", match_mode="bad")

    with pytest.raises(ValueError, match="run_mode.*both"):
        SaveAna(obj_type="particle", run_mode="reco", match_mode="both")

    with pytest.raises(ValueError, match="object types"):
        SaveAna(obj_type=None, match_mode="both")

    with pytest.raises(ValueError, match="not found"):
        SaveAna(obj_type="particle", particle=["definitely_missing"])


def test_save_ana_requires_at_least_one_writer(monkeypatch):
    monkeypatch.setattr(SaveAna, "initialize_writer", lambda self, name: None)

    with pytest.raises(ValueError, match="save something"):
        SaveAna(obj_type="particle", run_mode="reco", match_mode=None)


def test_save_ana_writes_objects_without_matches(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SaveAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    ana = SaveAna(
        obj_type="particle",
        particle=("id", "value"),
        run_mode="reco",
        match_mode=None,
    )

    ana.process({"reco_particles": [FakeObject(1, 3.0), FakeObject(2, 4.0)]})

    assert rows == [
        ("reco_particles", {"id": 1, "value": 3.0}),
        ("reco_particles", {"id": 2, "value": 4.0}),
    ]


def test_save_ana_writes_matched_objects(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SaveAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    ana = SaveAna(obj_type="particle", particle=("id",), match_mode="both")
    reco = FakeObject(1, 3.0)
    truth = FakeObject(2, 4.0)

    ana.process(
        {
            "particle_matches_r2t": [(reco, truth)],
            "particle_matches_r2t_overlap": [0.8],
            "particle_matches_t2r": [(truth, reco)],
            "particle_matches_t2r_overlap": [0.6],
        }
    )

    assert rows[0] == (
        "reco_particles",
        {
            "reco_id": 1,
            "truth_id": 2,
            "match_overlap": 0.8,
        },
    )
    assert rows[1][0] == "truth_particles"
    assert rows[1][1]["match_overlap"] == 0.6


def test_save_ana_uses_default_object_for_missing_match(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SaveAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    ana = SaveAna(obj_type="particle", particle=("id",), match_mode="reco_to_truth")
    reco = FakeObject(1, 3.0)

    ana.process(
        {
            "truth_particles": [],
            "particle_matches_r2t": [(reco, None)],
            "particle_matches_r2t_overlap": [0.2],
        }
    )

    assert rows == [
        (
            "reco_particles",
            {"reco_id": 1, "truth_id": -1, "match_overlap": 0.2},
        )
    ]
