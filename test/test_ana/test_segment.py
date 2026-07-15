from __future__ import annotations

import numpy as np
import pytest

from spine.ana.metric.segment import SegmentAna
from spine.constants import GHOST_SHP, SHAPE_COL


class FakeObject:
    def __init__(self, index, shape):
        self.index = np.asarray(index, dtype=np.int32)
        self.shape = shape


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(SegmentAna, "initialize_writer", lambda self, name: None)


def test_segment_ana_validates_configuration():
    with pytest.raises(ValueError, match="both fragments and particles"):
        SegmentAna(use_fragments=True, use_particles=True)

    with pytest.raises(ValueError, match="detailed score"):
        SegmentAna(use_fragments=True, summary=False)

    with pytest.raises(ValueError, match="ghost metrics"):
        SegmentAna(use_fragments=True, ghost=True)


def test_segment_ana_writes_summary(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SegmentAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    ana = SegmentAna(summary=True, num_classes=2)
    labels = np.zeros((3, 5), dtype=np.float32)
    labels[:, SHAPE_COL] = [0, 1, 1]

    ana.process(
        {
            "seg_label": labels,
            "segmentation": np.array([[4.0, 0.0], [0.0, 3.0], [2.0, 0.0]]),
        }
    )

    assert rows == [{"count_00": 1, "count_01": 1, "count_10": 0, "count_11": 1}]


def test_segment_ana_detailed_ghost_scores(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SegmentAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    ana = SegmentAna(summary=False, ghost=True)
    labels = np.zeros((2, 5), dtype=np.float32)
    labels[:, SHAPE_COL] = [0, GHOST_SHP]

    ana.process(
        {
            "seg_label": labels,
            "segmentation": np.array([[3.0, 0.0, 0.0, 0.0, 0.0]]),
            "ghost": np.array([[3.0, 0.0], [0.0, 3.0]]),
        }
    )

    assert len(rows) == 2
    assert rows[0]["score_0"] > 0.0
    assert rows[0][f"score_{GHOST_SHP}"] > 0.0


def test_segment_ana_rebuilds_summary_from_objects(monkeypatch):
    rows = []
    monkeypatch.setattr(
        SegmentAna, "append", lambda self, name, **kwargs: rows.append(kwargs)
    )
    ana = SegmentAna(use_fragments=True, num_classes=2)

    ana.process(
        {
            "points": np.zeros((3, 3)),
            "truth_fragments": [FakeObject([0, 1], 0), FakeObject([2], 1)],
            "reco_fragments": [FakeObject([0], 0), FakeObject([1, 2], 1)],
        }
    )

    assert rows[0]["count_00"] == 1
    assert rows[0]["count_10"] == 1
    assert rows[0]["count_11"] == 1


def test_segment_ana_rejects_empty_truth_object_index():
    ana = SegmentAna(use_particles=True)

    with pytest.raises(ValueError, match="index"):
        ana.process(
            {
                "points": np.zeros((1, 3)),
                "truth_particles": [FakeObject([], 0)],
                "reco_particles": [],
            }
        )


def test_segment_ana_rejects_missing_object_collection_state():
    ana = SegmentAna(use_particles=True)
    ana.object_collection = None

    with pytest.raises(ValueError, match="Object collection"):
        ana.process({"points": np.zeros((1, 3))})


def test_segment_ana_rejects_missing_scores_for_detailed_object_mode():
    ana = SegmentAna(use_particles=True)
    ana.summary = False

    with pytest.raises(ValueError, match="Segment scores"):
        ana.process(
            {
                "points": np.zeros((1, 3)),
                "truth_particles": [FakeObject([0], 0)],
                "reco_particles": [FakeObject([0], 0)],
            }
        )
