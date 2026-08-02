"""Tests for image classification and regression task losses."""

import math

import pytest

torch = pytest.importorskip("torch")

from spine.constants import ANCST_COL, ANCST_MOM_COL, ANCST_PID_COL, PID_MASSES
from spine.data import TensorBatch
from spine.model.image.loss import ImageLoss
from spine.model.image.object import ImageObjectBuilder


def test_classification_loss_reports_classwise_metrics(image_data):
    """Classification tasks should expose global and per-class accuracy."""
    objects = ImageObjectBuilder()(image_data)
    logits = TensorBatch(
        torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], requires_grad=True),
        counts=torch.tensor([1, 1]),
    )
    loss_fn = ImageLoss(
        {"heads": {"pid": 3}},
        {"pid": {"name": "class", "label": "labels"}},
    )

    result = loss_fn(objects, labels=[1, 2], pid_pred=logits)

    assert result["accuracy"] == 1.0
    assert result["pid_count"] == 2
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


def test_regression_loss_reports_physical_error_metrics(image_data):
    """Regression tasks should report residual metrics, not accuracy."""
    objects = ImageObjectBuilder()(image_data)
    predictions = TensorBatch(
        torch.tensor([[1.5], [3.0]], requires_grad=True),
        counts=torch.tensor([1, 1]),
    )
    loss_fn = ImageLoss(
        {"heads": {"energy": 1}},
        {
            "energy": {
                "name": "reg",
                "label": "labels",
                "loss": "huber",
            }
        },
    )

    result = loss_fn(objects, labels=[1.0, 2.0], energy_pred=predictions)

    assert "accuracy" not in result
    assert result["energy_bias"] == pytest.approx(0.75)
    assert result["energy_mae"] == pytest.approx(0.75)
    assert result["energy_rmse"] == pytest.approx((0.625) ** 0.5)
    result["loss"].backward()


def test_voxel_labels_are_reduced_over_cluster_objects(image_data):
    """Cluster tasks should derive targets using the configured label column."""
    objects = ImageObjectBuilder(source="cluster")(
        image_data,
        object_data=image_data,
    )
    logits = TensorBatch(
        torch.zeros((4, 5), requires_grad=True),
        counts=objects.counts,
    )
    loss_fn = ImageLoss(
        {"heads": {"pid": 5}},
        {
            "pid": {
                "name": "class",
                "label": "clust_label",
                "target": "pid",
            }
        },
    )

    result = loss_fn(objects, clust_label=image_data, pid_pred=logits)

    assert result["pid_count"] == 4
    assert torch.isfinite(result["loss"])


def test_ancestor_targets_use_root_particle_pid(image_data):
    """Ancestor PID must come from the root rather than the modal descendant."""
    labels = image_data.torch_tensor().clone()
    labels[:4, ANCST_COL] = 0
    labels[:4, ANCST_PID_COL] = 2
    labels[:4, ANCST_MOM_COL] = 200
    ancestor_data = TensorBatch(labels, image_data.counts)
    objects = ImageObjectBuilder(source="ancestor")(
        ancestor_data,
        object_data=ancestor_data,
    )
    logits = TensorBatch(
        torch.zeros((len(objects.index_list), 5), requires_grad=True),
        counts=objects.counts,
    )
    loss_fn = ImageLoss(
        {"heads": {"pid": 5}},
        {
            "pid": {
                "name": "class",
                "label": "clust_label",
                "target": "pid",
                "target_reduction": "ancestor",
            }
        },
    )

    result = loss_fn(objects, clust_label=ancestor_data, pid_pred=logits)

    assert result["pid_count"] == 3
    assert torch.isfinite(result["loss"])


def test_ancestor_energy_is_derived_from_root_momentum(image_data):
    """Ancestor energy should be the root particle's initial kinetic energy."""
    labels = image_data.torch_tensor().clone()
    labels[:4, ANCST_COL] = 0
    labels[:4, ANCST_PID_COL] = 2
    labels[:4, ANCST_MOM_COL] = 200
    ancestor_data = TensorBatch(labels, image_data.counts)
    objects = ImageObjectBuilder(source="ancestor")(
        ancestor_data,
        object_data=ancestor_data,
    )
    predictions = TensorBatch(
        torch.zeros((len(objects.index_list), 1), requires_grad=True),
        counts=objects.counts,
    )
    loss_fn = ImageLoss(
        {"heads": {"energy": 1}},
        {
            "energy": {
                "name": "reg",
                "label": "clust_label",
                "target": "kinetic_energy",
                "target_reduction": "ancestor",
            }
        },
    )

    result = loss_fn(
        objects,
        clust_label=ancestor_data,
        energy_pred=predictions,
    )

    momenta = (200.0, 10.0, 500.0)
    particle_ids = (2, 1, 4)
    expected_mae = sum(
        math.sqrt(momentum**2 + PID_MASSES[pid] ** 2) - PID_MASSES[pid]
        for momentum, pid in zip(momenta, particle_ids, strict=True)
    ) / len(momenta)
    assert result["energy_count"] == 3
    assert result["energy_mae"] == pytest.approx(expected_mae)
    assert torch.isfinite(result["loss"])
