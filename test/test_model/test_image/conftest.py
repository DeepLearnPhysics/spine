"""Shared fixtures for image-model tests."""

import pytest

torch = pytest.importorskip("torch")

from spine.constants import (
    ANCST_COL,
    ANCST_MOM_COL,
    ANCST_PID_COL,
    CLUST_COL,
    GROUP_COL,
    MOM_COL,
    PART_COL,
    PID_COL,
    SHAPE_COL,
    VALUE_COL,
    ClusterLabelCol,
)
from spine.data import TensorBatch


@pytest.fixture
def image_data():
    """Return two sparse events with cluster, group, shape, and PID labels."""
    width = int(ClusterLabelCol.SHAPE) + 1
    data = torch.zeros((7, width), dtype=torch.float32)
    data[:, 0] = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    data[:, 1:4] = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
        ]
    )
    data[:, VALUE_COL] = torch.arange(1, 8)
    data[:, CLUST_COL] = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    data[:, PART_COL] = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    data[:, GROUP_COL] = torch.tensor([0, 0, 0, 0, 0, 0, 1])
    data[:, ANCST_COL] = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    data[:, SHAPE_COL] = 1
    data[:, PID_COL] = torch.tensor([2, 2, 3, 3, 1, 1, 4])
    data[:, MOM_COL] = torch.tensor([200, 200, 300, 300, 10, 10, 500])
    data[:, ANCST_PID_COL] = data[:, PID_COL]
    data[:, ANCST_MOM_COL] = data[:, MOM_COL]
    return TensorBatch(data, counts=torch.tensor([4, 3]))
