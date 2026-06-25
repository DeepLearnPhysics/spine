"""Tests for clustering analysis metrics."""

import numpy as np

from spine.ana.metric.cluster import ClusterAna
from spine.data.out import RecoParticle, TruthParticle


def _run_particle_cluster_ana(monkeypatch, truth_index_mode="index_adapt"):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )

    ana = ClusterAna(
        obj_type="particle",
        use_objects=True,
        per_shape=False,
        metrics=("ari",),
        truth_index_mode=truth_index_mode,
    )
    ana.process(
        {
            "points": np.zeros((4, 3), dtype=np.float32),
            "truth_particles": [
                TruthParticle(
                    id=0,
                    index=np.array([0, 1], dtype=np.int32),
                    index_adapt=np.array([0, 2], dtype=np.int32),
                ),
                TruthParticle(
                    id=1,
                    index=np.array([2, 3], dtype=np.int32),
                    index_adapt=np.array([1, 3], dtype=np.int32),
                ),
            ],
            "reco_particles": [
                RecoParticle(id=0, index=np.array([0, 1], dtype=np.int32)),
                RecoParticle(id=1, index=np.array([2, 3], dtype=np.int32)),
            ],
        }
    )

    return rows[0]


def test_cluster_ana_truth_index_mode_selects_truth_object_index(monkeypatch):
    row = _run_particle_cluster_ana(monkeypatch, truth_index_mode="index")

    assert row["num_points"] == 4
    assert row["num_truth"] == 2
    assert row["num_reco"] == 2
    assert row["ari"] == 1.0


def test_cluster_ana_truth_index_mode_defaults_to_index_adapt(monkeypatch):
    row = _run_particle_cluster_ana(monkeypatch)

    assert row["ari"] != 1.0
