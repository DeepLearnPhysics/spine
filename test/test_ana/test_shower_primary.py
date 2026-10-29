"""Training/report agreement for shower-primary targets and eligibility."""

import numpy as np
import pytest

from spine.ana.metric.shower_primary import ShowerPrimaryAna
from spine.cluster.primary import primary_purity_mask_batch, shower_primary_target
from spine.data import ClusterLabelData, IndexBatch, TensorBatch, TensorData
from spine.io.collate import CollateAll


def event():
    """Split the primary particle into two fragments in one shower."""
    labels = ClusterLabelData(
        coords=np.array([[5, 0, 0], [1, 0, 0], [9, 0, 0]], dtype=np.float32),
        features=np.array([[1, 0, 0], [1, 1, 0], [1, 2, 1]], dtype=np.float32),
        particles={
            "particle": np.array([0, 1]),
            "group": np.array([0, 0]),
            "group_primary": np.array([1, 0]),
        },
    )
    coordinates = TensorData(
        coords=np.array([[0, 0, 0], [8, 0, 0]], dtype=np.float32),
        features=np.zeros((2, 1), dtype=np.float32),
        coordinate_groups={"start": (0, 1, 2)},
    )
    return {
        "clust_label_adapt": labels,
        "coord_label": coordinates,
        "fragment_clusts": [np.array([i]) for i in range(3)],
        "fragment_shapes": np.zeros(3, dtype=int),
        "fragment_group_pred": np.zeros(3, dtype=int),
        "fragment_node_pred": np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
    }


@pytest.mark.parametrize(
    "groups,expected", [([0, 0, 0], [True] * 3), ([0, 1, 0], [False, True, False])]
)
def test_report_closest_target_and_predicted_group_eligibility(
    monkeypatch, groups, expected
):
    """Closest selection resolves split primaries before group-purity cuts."""
    rows = []
    monkeypatch.setattr(ShowerPrimaryAna, "initialize_writer", lambda *_: None)
    monkeypatch.setattr(
        ShowerPrimaryAna, "append", lambda _, name, **row: rows.append(row)
    )
    data = event()
    data["fragment_group_pred"] = np.array(groups)
    analyzer = ShowerPrimaryAna(use_closest=True, high_purity=True, use_group_pred=True)
    analyzer.process(data)
    assert [row["target"] for row in rows] == [0, 1, 0]
    assert [row["valid"] for row in rows] == expected


def test_report_empty_shower_collection(monkeypatch):
    """Empty shower events produce no fabricated negative examples."""
    monkeypatch.setattr(ShowerPrimaryAna, "initialize_writer", lambda *_: None)
    data = event()
    data["fragment_shapes"][:] = 1
    ShowerPrimaryAna().process(data)


@pytest.mark.parametrize("quality", [False, True])
def test_report_preserves_group_target_and_explicit_eligibility(monkeypatch, quality):
    """Unresolved split primaries and overlap rejection remain inspectable."""
    rows = []
    monkeypatch.setattr(ShowerPrimaryAna, "initialize_writer", lambda *_: None)
    monkeypatch.setattr(
        ShowerPrimaryAna, "append", lambda _, name, **row: rows.append(row)
    )
    analyzer = ShowerPrimaryAna(min_efficiency=0.9 if quality else None)
    analyzer.process(event())
    assert [row["target"] for row in rows] == [1, 1, 0]
    assert [row["valid"] for row in rows] == [not quality] * 3
    assert {row["target_definition"] for row in rows} == {"group_primary"}


def test_report_keeps_michel_and_delta_nodes_in_target_domain(monkeypatch):
    """Purity/closest selection sees the complete shower-network node domain."""
    rows = []
    monkeypatch.setattr(ShowerPrimaryAna, "initialize_writer", lambda *_: None)
    monkeypatch.setattr(
        ShowerPrimaryAna, "append", lambda _, name, **row: rows.append(row)
    )
    data = event()
    data["fragment_shapes"] = np.array([0, 2, 3])
    ShowerPrimaryAna(use_closest=True, high_purity=True).process(data)
    assert [row["target"] for row in rows] == [0, 1, 0]
    assert all(row["valid"] for row in rows)


def test_shared_targets_reject_missing_coordinates_and_invalid_truth():
    """Closest targets require start points; unknown truth stays ineligible."""
    data = event()
    labels = CollateAll(data_keys=("labels",))([{"labels": data["clust_label_adapt"]}])[
        "labels"
    ]
    clusters = IndexBatch(
        data["fragment_clusts"], spans=[3], counts=[3], single_counts=[1, 1, 1]
    )
    with pytest.raises(ValueError, match="coord_label"):
        shower_primary_target(labels, clusters, use_closest=True)
    labels.particles["group_primary"].data[0] = -1
    target, valid = shower_primary_target(labels, clusters)
    assert target.data.tolist() == [-1, -1, 0]
    assert valid.tolist() == [False, False, True]


def test_shared_purity_keeps_event_groups_separate():
    """Repeated group IDs across events must not combine primary counts."""
    groups = TensorBatch(np.array([0, 0, 1, 0, 0]), [3, 2])
    target = TensorBatch(np.array([1, 0, -1, 1, 1]), [3, 2])
    assert primary_purity_mask_batch(groups, target).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_unmodeled_nodes_do_not_steal_closest_primary(monkeypatch):
    """A filtered node's sentinel logits exclude it before target creation."""
    rows = []
    monkeypatch.setattr(ShowerPrimaryAna, "initialize_writer", lambda *_: None)
    monkeypatch.setattr(
        ShowerPrimaryAna, "append", lambda _, name, **row: rows.append(row)
    )
    data = event()
    data["fragment_node_pred"][1] = -np.inf
    ShowerPrimaryAna(use_closest=True, high_purity=True).process(data)
    assert [row["fragment_id"] for row in rows] == [0, 2]
    assert [row["target"] for row in rows] == [1, 0]
    assert all(row["valid"] for row in rows)
