from __future__ import annotations

import numpy as np
import pytest

from spine.ana.metric.point import PointProposalAna
from spine.data import TensorData, TensorSchema
from spine.utils.ppn import ppn_prediction_schema


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(PointProposalAna, "initialize_writer", lambda self, name: None)


def _point_label(
    coords: list[list[float]], shapes: list[int], endpoints: list[int] | None = None
) -> TensorData:
    features = np.zeros((len(coords), 3), dtype=np.float32)
    features[:, 0] = shapes
    if endpoints is not None:
        features[:, 2] = endpoints
    return TensorData(
        coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        features=features,
        schema=TensorSchema(
            coordinate_groups={"point": (0, 1, 2)},
            feature_fields={"shape": (0,), "particle": (1,), "endpoint": (2,)},
        ),
    )


def _point_prediction(
    coords: list[list[float]],
    shapes: list[int],
    endpoint_scores: list[list[float]] | None = None,
) -> TensorData:
    features = np.zeros(
        (len(coords), 9 + 2 * (endpoint_scores is not None)), dtype=np.float32
    )
    features[:, 8] = shapes
    if endpoint_scores is not None:
        features[:, 9:11] = np.asarray(endpoint_scores).reshape(-1, 2)
    return TensorData(
        coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        features=features,
        schema=ppn_prediction_schema(endpoint_scores is not None),
    )


def test_point_proposal_ana_processes_bidirectional_matches(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointProposalAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    labels = _point_label([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [0, 1], [0, 1])
    preds = _point_prediction([[1.0, 0.0, 0.0]], [0], [[0.1, 0.9]])
    ana = PointProposalAna(num_classes=2, endpoints=True)

    ana.process({"ppn_label": labels, "ppn_pred": preds})

    names = [name for name, _ in rows]
    assert names == ["truth_to_reco", "truth_to_reco", "reco_to_truth"]
    assert rows[0][1]["dist"] == 1.0
    assert rows[0][1]["closest_end"] == 1
    assert rows[-1][1]["shape"] == 0


def test_point_proposal_ana_records_dummy_when_target_is_empty(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointProposalAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    labels = _point_label([[0.0, 0.0, 0.0]], [1])
    preds = _point_prediction([], [])
    ana = PointProposalAna(num_classes=2)

    ana.process({"ppn_label": labels, "ppn_pred": preds})

    assert rows == [
        (
            "truth_to_reco",
            {
                "dist": -1.0,
                "shape": 1,
                "closest_shape": -1,
                "dist_0": -1.0,
                "dist_1": -1.0,
            },
        )
    ]


def test_point_proposal_ana_records_endpoint_dummy_when_target_is_empty(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointProposalAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    labels = _point_label([[0.0, 0.0, 0.0]], [1], [0])
    preds = _point_prediction([], [], [])
    ana = PointProposalAna(num_classes=2, endpoints=True)

    ana.process({"ppn_label": labels, "ppn_pred": preds})

    assert rows[0][1]["end"] == 0
    assert rows[0][1]["closest_end"] == -1
