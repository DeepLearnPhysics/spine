"""Tests for clustering analysis metrics."""

import numpy as np
import pytest

from spine.ana.metric.cluster import ClusterAna
from spine.data import ClusterLabelData
from spine.data.out import (
    RecoFragment,
    RecoInteraction,
    RecoParticle,
    TruthFragment,
    TruthInteraction,
    TruthParticle,
)


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


def test_cluster_ana_validates_configuration(monkeypatch):
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)

    with pytest.raises(ValueError, match="provide a list"):
        ClusterAna(obj_type=None, per_object=True)

    with pytest.raises(ValueError, match="target clustering label"):
        ClusterAna(per_object=False)

    with pytest.raises(ValueError, match="cannot.*use objects"):
        ClusterAna(per_object=False, label_col="group", use_objects=True)

    with pytest.raises(ValueError, match="two values"):
        ClusterAna(obj_type="particle", time_window=(0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="lower bound"):
        ClusterAna(obj_type="particle", time_window=(1.0, 0.0))

    with pytest.raises(ValueError, match="standalone mode"):
        ClusterAna(
            per_object=False,
            label_col="unsupported",
            time_window=(0.0, 1.0),
        )


def test_cluster_ana_raw_per_object_mode(monkeypatch):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    labels = ClusterLabelData(
        np.asarray(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [2, 0, 0, 1, 1, 1],
                [3, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        {"shape": np.asarray([0, 1])},
    )
    ana = ClusterAna(
        obj_type="fragment",
        use_objects=False,
        per_shape=True,
        metrics=("ari",),
    )

    ana.process(
        {
            "clust_label_adapt": labels,
            "fragment_clusts": [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ],
            "fragment_shapes": [0, 1],
        }
    )

    assert rows[0][0] == "fragment"
    assert rows[0][1]["ari"] == 1.0
    assert rows[0][1]["ari_0"] == 1.0
    assert rows[0][1]["ari_1"] == 1.0


def test_cluster_ana_time_filters_raw_per_object_labels(monkeypatch):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    labels = ClusterLabelData(
        np.asarray(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [2, 0, 0, 1, 1, 1],
                [3, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        {"group": np.asarray([0, 1])},
    )
    labels_before = labels.data.copy()
    ana = ClusterAna(
        obj_type="particle",
        use_objects=False,
        per_shape=False,
        metrics=("ari",),
        time_window=(0.0, 10.0),
    )

    assert ana.keys["truth_particles"] is True
    ana.process(
        {
            "clust_label_adapt": labels,
            "truth_particles": [
                TruthParticle(
                    t=5.0,
                    index_adapt=np.array([0, 1], dtype=np.int32),
                ),
                TruthParticle(
                    t=20.0,
                    index_adapt=np.array([2, 3], dtype=np.int32),
                ),
            ],
            "particle_clusts": [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ],
            "particle_shapes": [0, 1],
        }
    )

    assert rows == [{"num_points": 4, "num_truth": 1, "num_reco": 2, "ari": 1.0}]
    np.testing.assert_array_equal(labels.data, labels_before)


def test_cluster_ana_standalone_mode(monkeypatch):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    labels = ClusterLabelData(
        np.asarray(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [2, 0, 0, 1, 1, 1],
                [3, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        {"group": np.asarray([0, 1])},
    )
    ana = ClusterAna(
        per_object=False,
        per_shape=False,
        label_col="group",
        metrics=("ari",),
    )

    ana.process(
        {
            "clust_label_adapt": labels,
            "clusts": [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ],
            "group_pred": [0, 1],
        }
    )

    assert rows == [
        (
            "group",
            {
                "num_points": 4,
                "num_truth": 2,
                "num_reco": 2,
                "ari": 1.0,
            },
        )
    ]


def test_cluster_ana_time_filters_standalone_labels(monkeypatch):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    labels = ClusterLabelData(
        np.asarray(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [2, 0, 0, 1, 1, 1],
                [3, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        {"group": np.asarray([0, 1])},
    )
    ana = ClusterAna(
        per_object=False,
        per_shape=False,
        label_col="group",
        metrics=("ari",),
        time_window=(0.0, 10.0),
    )

    assert ana.keys["truth_particles"] is True
    ana.process(
        {
            "clust_label_adapt": labels,
            "truth_particles": [
                TruthParticle(
                    t=5.0,
                    index_adapt=np.array([0, 1], dtype=np.int32),
                ),
                TruthParticle(
                    t=20.0,
                    index_adapt=np.array([2, 3], dtype=np.int32),
                ),
            ],
            "clusts": [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ],
            "group_pred": [0, 1],
        }
    )

    assert rows == [{"num_points": 4, "num_truth": 1, "num_reco": 2, "ari": 1.0}]


def test_cluster_ana_does_not_produce_per_shape_interaction_metrics(monkeypatch):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    ana = ClusterAna(
        obj_type="interaction",
        use_objects=True,
        per_shape=True,
        metrics=("ari",),
        truth_index_mode="index",
    )

    ana.process(
        {
            "points": np.zeros((2, 3), dtype=np.float32),
            "truth_interactions": [
                TruthInteraction(index=np.array([0, 1], dtype=np.int32))
            ],
            "reco_interactions": [
                RecoInteraction(index=np.array([0, 1], dtype=np.int32))
            ],
        }
    )

    assert rows == [{"num_points": 2, "num_truth": 1, "num_reco": 1, "ari": 1.0}]


@pytest.mark.parametrize("obj_type", ("fragment", "particle", "interaction"))
def test_cluster_ana_time_filters_truth_objects(monkeypatch, obj_type):
    rows = []
    monkeypatch.setattr(ClusterAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        ClusterAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    ana = ClusterAna(
        obj_type=obj_type,
        use_objects=True,
        per_shape=False,
        metrics=("ari",),
        truth_index_mode="index",
        time_window=(0.0, 10.0),
    )

    truth_type = {
        "fragment": TruthFragment,
        "particle": TruthParticle,
        "interaction": TruthInteraction,
    }[obj_type]
    reco_type = {
        "fragment": RecoFragment,
        "particle": RecoParticle,
        "interaction": RecoInteraction,
    }[obj_type]
    truth_objects = [
        truth_type(index=np.array([0, 1], dtype=np.int32), t=5.0),
        truth_type(index=np.array([2, 3], dtype=np.int32), t=20.0),
    ]
    reco_objects = [
        reco_type(index=np.array([0, 1], dtype=np.int32)),
        reco_type(index=np.array([2, 3], dtype=np.int32)),
    ]

    ana.process(
        {
            "points": np.zeros((4, 3), dtype=np.float32),
            f"truth_{obj_type}s": truth_objects,
            f"reco_{obj_type}s": reco_objects,
        }
    )

    assert rows == [{"num_points": 4, "num_truth": 1, "num_reco": 2, "ari": 1.0}]
