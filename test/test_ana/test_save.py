from __future__ import annotations

import numpy as np
import pytest

from spine.ana.script.save import SaveAna
from spine.data import RecoParticle, TruthParticle


class FakeWriter:
    def __init__(self):
        self.columns = []

    def append_columns(self, data):
        self.columns.append(data)

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

    with pytest.raises(ValueError, match="requires.*truth.*source"):
        SaveAna(
            obj_type="particle",
            run_mode="reco",
            match_mode="truth_to_reco",
        )

    with pytest.raises(ValueError, match="object types"):
        SaveAna(obj_type=None, match_mode="both")

    with pytest.raises(ValueError, match="not found"):
        SaveAna(obj_type="particle", particle=["definitely_missing"])


def test_save_ana_requires_at_least_one_writer(monkeypatch):
    monkeypatch.setattr(SaveAna, "initialize_writer", lambda self, name: None)

    with pytest.raises(ValueError, match="save something"):
        SaveAna(obj_type="particle", run_mode="reco", match_mode=None)


def test_save_ana_writes_objects_without_matches(monkeypatch):
    ana = SaveAna(
        obj_type="particle",
        particle=("pid", "id"),
        run_mode="reco",
        match_mode=None,
    )

    ana.process(
        {
            "reco_particles": [
                RecoParticle(id=1, pid=3),
                RecoParticle(id=2, pid=4),
            ]
        }
    )

    columns = ana.writers["reco_particles"].columns
    assert len(columns) == 1
    assert list(columns[0]) == ["id", "pid"]
    assert columns[0]["id"].tolist() == [1, 2]
    assert columns[0]["pid"].tolist() == [3, 4]


def test_save_ana_writes_matched_objects(monkeypatch):
    ana = SaveAna(obj_type="particle", particle=("id",), match_mode="both")
    reco = RecoParticle(id=1)
    truth = TruthParticle(id=2)

    ana.process(
        {
            "particle_matches_r2t": [(reco, truth)],
            "particle_matches_r2t_overlap": [0.8],
            "particle_matches_t2r": [(truth, reco)],
            "particle_matches_t2r_overlap": [0.6],
        }
    )

    reco_columns = ana.writers["reco_particles"].columns[0]
    assert reco_columns["reco_id"].tolist() == [1]
    assert reco_columns["truth_id"].tolist() == [2]
    assert reco_columns["match_overlap"].tolist() == [0.8]
    truth_columns = ana.writers["truth_particles"].columns[0]
    assert truth_columns["match_overlap"].tolist() == [0.6]


def test_save_ana_uses_default_object_for_missing_match(monkeypatch):
    ana = SaveAna(obj_type="particle", particle=("id",), match_mode="reco_to_truth")
    reco = RecoParticle(id=1)

    ana.process(
        {
            "truth_particles": [],
            "particle_matches_r2t": [(reco, None)],
            "particle_matches_r2t_overlap": [0.2],
        }
    )

    columns = ana.writers["reco_particles"].columns[0]
    assert columns["reco_id"].tolist() == [1]
    assert columns["truth_id"].tolist() == [-1]
    assert columns["match_overlap"].tolist() == [0.2]


def test_save_ana_writes_only_truth_to_reco_source_rows():
    """Directional matching should load reco support without a reco writer."""
    ana = SaveAna(
        obj_type="particle",
        particle=("id", "pid"),
        run_mode="truth",
        match_mode="truth_to_reco",
    )
    truth = TruthParticle(id=4, pid=2)
    reco = RecoParticle(id=7, pid=3)

    assert list(ana.writers) == ["truth_particles"]
    assert ana.keys["reco_particles"] is True
    ana.process(
        {
            "particle_matches_t2r": [(truth, reco)],
            "particle_matches_t2r_overlap": [0.75],
        }
    )

    columns = ana.writers["truth_particles"].columns[0]
    assert columns["truth_id"].tolist() == [4]
    assert columns["reco_id"].tolist() == [7]
    assert columns["match_overlap"].tolist() == [0.75]


def test_save_ana_columnar_truth_to_reco_is_directional():
    """The projected path should request targets without writing target rows."""
    ana = SaveAna(
        obj_type="particle",
        particle=("id", "pid"),
        run_mode="truth",
        match_mode="truth_to_reco",
    )
    requests = ana.columnar_requests()
    assert set(requests) == {"run_info", "truth_particles", "reco_particles"}
    assert "best_match_id" in requests["truth_particles"][0]
    assert "best_match_id" not in requests["reco_particles"][0]

    ana.process_columnar(
        {
            "index": np.asarray([0]),
            "file_index": np.asarray([0]),
            "truth_particles": {
                "id": np.asarray([0, 1]),
                "pid": np.asarray([2, 4]),
                "best_match_id": np.asarray([0, -1]),
                "best_match_overlap": np.asarray([0.8, -1.0]),
                "event_offsets": np.asarray([0, 2]),
            },
            "reco_particles": {
                "id": np.asarray([3]),
                "pid": np.asarray([2]),
                "event_offsets": np.asarray([0, 1]),
            },
        }
    )

    assert list(ana.writers) == ["truth_particles"]
    columns = ana.writers["truth_particles"].columns[0]
    assert columns["truth_id"].tolist() == [0, 1]
    assert columns["reco_id"].tolist() == [3, -1]
    assert columns["match_overlap"].tolist() == [0.8, -1.0]


def test_save_ana_directional_event_falls_back_to_object_matches():
    """Directional event saving should consume stored best-match metadata."""
    ana = SaveAna(
        obj_type="particle",
        particle=("id",),
        run_mode="truth",
        match_mode="truth_to_reco",
    )
    truth = TruthParticle(
        id=4,
        is_matched=True,
        match_ids=np.asarray([0], dtype=np.int32),
        match_overlaps=np.asarray([0.625], dtype=np.float32),
    )
    ana.process(
        {
            "truth_particles": [truth],
            "reco_particles": [RecoParticle(id=7)],
        }
    )

    columns = ana.writers["truth_particles"].columns[0]
    assert columns["truth_id"].tolist() == [4]
    assert columns["reco_id"].tolist() == [7]
    assert columns["match_overlap"].tolist() == pytest.approx([0.625])


def test_save_ana_columnar_joins_best_matches():
    ana = SaveAna(
        obj_type="particle",
        particle=("id", "pid", "size"),
        match_mode="reco_to_truth",
    )
    data = {
        "index": np.asarray([0, 1]),
        "file_index": np.asarray([0, 0]),
        "file_entry_index": np.asarray([0, 1]),
        "reco_particles": {
            "id": np.asarray([0, 1, 0]),
            "pid": np.asarray([2, 4, 3]),
            "size": np.asarray([10, 20, 30]),
            "best_match_id": np.asarray([0, -1, 0]),
            "best_match_overlap": np.asarray([0.8, -1.0, 0.6]),
            "event_offsets": np.asarray([0, 2, 3]),
        },
        "truth_particles": {
            "id": np.asarray([0, 0]),
            "pid": np.asarray([2, 3]),
            "size": np.asarray([12, 28]),
            "event_offsets": np.asarray([0, 1, 2]),
        },
    }

    ana.process_columnar(data)

    reco = ana.writers["reco_particles"].columns[0]
    assert reco["index"].tolist() == [0, 0, 1]
    assert reco["reco_id"].tolist() == [0, 1, 0]
    assert reco["truth_pid"].tolist() == [2, -1, 3]
    assert reco["truth_size"].tolist() == [12, 0, 28]
    assert reco["match_overlap"].tolist() == [0.8, -1.0, 0.6]

    truth = ana.writers["truth_particles"].columns[0]
    assert truth["index"].tolist() == [0, 1]
    assert truth["id"].tolist() == [0, 0]


def test_save_ana_validates_columnar_attributes():
    """Columnar save requires explicit, fixed-width object attributes."""
    implicit = SaveAna(obj_type="particle", run_mode="reco", match_mode=None)
    with pytest.raises(ValueError, match="explicit attribute"):
        implicit.columnar_requests()

    variable = SaveAna(
        obj_type="particle",
        particle=("index",),
        run_mode="reco",
        match_mode=None,
    )
    with pytest.raises(ValueError, match="variable fields"):
        variable.columnar_requests()


def test_save_ana_expands_fixed_width_columnar_attributes():
    """Fixed vectors should expand into named scalar output columns."""
    product = {"start_point": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)}
    columns = SaveAna._expand_columnar_attrs(product, ("start_point",), RecoParticle())
    assert list(columns) == ["start_point_x", "start_point_y", "start_point_z"]

    with pytest.raises(ValueError, match="scalar or fixed-width"):
        SaveAna._expand_columnar_attrs(
            {"start_point": np.zeros((1, 2, 3))},
            ("start_point",),
            RecoParticle(),
        )


def test_save_ana_uses_bulk_object_columns():
    """Bulk object columns support the fields needed by event serialization."""
    objects = [
        RecoParticle(
            id=1,
            start_point=np.asarray([1.0, 2.0, 3.0]),
            pid_scores=np.arange(6, dtype=np.float32),
            match_ids=np.asarray([4], dtype=np.int32),
        ),
        RecoParticle(
            id=2,
            start_point=np.asarray([4.0, 5.0, 6.0]),
            pid_scores=np.arange(6, dtype=np.float32) + 6,
            match_ids=np.asarray([], dtype=np.int32),
        ),
    ]
    schema = RecoParticle()
    columns = schema.scalar_columns(
        objects,
        ("id", "start_point", "pid_scores", "match_ids"),
        {"match_ids": 2},
    )

    assert columns["id"].tolist() == [1, 2]
    assert columns["start_point_z"].tolist() == [3.0, 6.0]
    assert columns["pid_scores_5"].tolist() == [5.0, 11.0]
    assert columns["match_ids_0"].tolist() == [4, None]
    assert columns["match_ids_1"].tolist() == [None, None]

    implicit = schema.scalar_columns(
        objects,
        None,
        None,
    )
    assert "id" in implicit
    assert "match_ids_0" not in implicit

    with pytest.raises(ValueError, match="provide a fixed length"):
        schema.scalar_columns(
            objects,
            ("match_ids",),
            None,
        )
    with pytest.raises(AttributeError, match="do\(es\) not appear"):
        schema.scalar_columns(
            objects,
            ("missing",),
            None,
        )
