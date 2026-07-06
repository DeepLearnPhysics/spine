from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import GHOST_SHP
from spine.post.truth.label import ChildrenProcessor


def test_children_processor_counts_shape_children():
    parent = SimpleNamespace(
        orig_id=10,
        parent_id=10,
        shape=1,
        children_counts=np.zeros(GHOST_SHP, dtype=np.int32),
    )
    child_a = SimpleNamespace(
        orig_id=11,
        parent_id=10,
        shape=2,
        children_counts=np.zeros(GHOST_SHP, dtype=np.int32),
    )
    child_b = SimpleNamespace(
        orig_id=12,
        parent_id=10,
        shape=2,
        children_counts=np.zeros(GHOST_SHP, dtype=np.int32),
    )
    processor = ChildrenProcessor(mode="shape")

    processor.process({"truth_particles": [parent, child_a, child_b]})

    assert parent.children_counts[2] == 2
    assert child_a.children_counts.sum() == 0


def test_children_processor_counts_pid_children():
    parent = SimpleNamespace(
        orig_id=10,
        parent_id=10,
        pid=1,
        children_counts=np.zeros(6, dtype=np.int32),
    )
    child = SimpleNamespace(
        orig_id=11,
        parent_id=10,
        pid=2,
        children_counts=np.zeros(6, dtype=np.int32),
    )
    processor = ChildrenProcessor(mode="pid")

    processor.process({"truth_particles": [parent, child]})

    assert parent.children_counts[2] == 1


def test_children_processor_validates_mode():
    with pytest.raises(ValueError, match="not recognized"):
        ChildrenProcessor(mode="bad")
