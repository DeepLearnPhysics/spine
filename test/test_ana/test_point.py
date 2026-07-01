from __future__ import annotations

import numpy as np
import pytest

from spine.ana.metric.point import PointProposalAna
from spine.constants import (
    COORD_COLS,
    PPN_END_COLS,
    PPN_LENDP_COL,
    PPN_LTYPE_COL,
    PPN_SHAPE_COL,
)


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(PointProposalAna, "initialize_writer", lambda self, name: None)


def _point_tensor(num_rows: int) -> np.ndarray:
    return np.zeros((num_rows, max(PPN_END_COLS) + 1), dtype=np.float32)


def test_point_proposal_ana_processes_bidirectional_matches(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointProposalAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    labels = _point_tensor(2)
    labels[:, COORD_COLS] = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    labels[:, PPN_LTYPE_COL] = [0, 1]
    labels[:, PPN_LENDP_COL] = [0, 1]
    preds = _point_tensor(1)
    preds[:, COORD_COLS] = [[1.0, 0.0, 0.0]]
    preds[:, PPN_SHAPE_COL] = [0]
    preds[:, PPN_END_COLS] = [[0.1, 0.9]]
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
    labels = _point_tensor(1)
    labels[:, COORD_COLS] = [[0.0, 0.0, 0.0]]
    labels[:, PPN_LTYPE_COL] = [1]
    preds = _point_tensor(0)
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
    labels = _point_tensor(1)
    labels[:, COORD_COLS] = [[0.0, 0.0, 0.0]]
    labels[:, PPN_LTYPE_COL] = [1]
    labels[:, PPN_LENDP_COL] = [0]
    preds = _point_tensor(0)
    ana = PointProposalAna(num_classes=2, endpoints=True)

    ana.process({"ppn_label": labels, "ppn_pred": preds})

    assert rows[0][1]["end"] == 0
    assert rows[0][1]["closest_end"] == -1
