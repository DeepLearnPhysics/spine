"""Tests for image classification and regression task losses."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spine.constants import PID_MASSES
from spine.data import ClusterLabelBatch, TensorBatch
from spine.model.image.loss import (
    ImageClassificationLoss,
    ImageLoss,
    ImageRegressionLoss,
)
from spine.model.image.object import ImageObjectBuilder


def _with_shared_first_ancestor(image_data):
    """Point both particles in the first event at its first particle."""
    particles = {
        name: TensorBatch(field.torch_tensor().clone(), field.counts)
        for name, field in image_data.particles.items()
    }
    particles["ancestor"].data[:2] = 0
    data = TensorBatch(
        image_data.torch_tensor().clone(),
        image_data.counts,
        has_batch_col=True,
    )
    return ClusterLabelBatch(data, particles)


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
    assert "pid_count_rejected" not in result
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


def test_image_losses_filter_low_quality_objects(image_data):
    """Image classification and regression use identical overlap gates."""
    objects = ImageObjectBuilder(source="cluster")(
        image_data,
        object_data=image_data,
    )
    logits = TensorBatch(torch.zeros((4, 5)), counts=objects.counts)
    class_result = ImageClassificationLoss(
        5,
        target="pid",
        min_efficiency=0.75,
    )(image_data, objects, logits)
    assert class_result["count"] == 2
    assert class_result["count_rejected"] == 2

    predictions = TensorBatch(torch.zeros((4, 1)), counts=objects.counts)
    reg_result = ImageRegressionLoss(
        1,
        target="momentum",
        min_efficiency=0.75,
    )(image_data, objects, predictions)
    assert reg_result["count"] == 2
    assert reg_result["count_rejected"] == 2


def test_image_losses_report_when_quality_rejects_every_object(
    image_data,
    monkeypatch,
):
    """Zero-count objectives should retain their quality rejection count."""
    objects = ImageObjectBuilder(source="cluster")(
        image_data,
        object_data=image_data,
    )

    class_loss = ImageClassificationLoss(5, target="pid", min_iou=0.0)
    monkeypatch.setattr(
        class_loss.quality_filter,
        "node_mask",
        lambda *args: np.zeros(len(objects.index_list), dtype=bool),
    )
    logits = TensorBatch(torch.zeros((4, 5)), counts=objects.counts)
    class_result = class_loss(image_data, objects, logits)
    assert class_result["count"] == 0
    assert class_result["count_rejected"] == 4

    reg_loss = ImageRegressionLoss(1, target="momentum", min_iou=0.0)
    monkeypatch.setattr(
        reg_loss.quality_filter,
        "node_mask",
        lambda *args: np.zeros(len(objects.index_list), dtype=bool),
    )
    predictions = TensorBatch(torch.zeros((4, 1)), counts=objects.counts)
    reg_result = reg_loss(image_data, objects, predictions)
    assert reg_result["count"] == 0
    assert reg_result["count_rejected"] == 4


def test_image_overlap_thresholds_validate_inputs(image_data):
    """Image overlap gates require structured labels and valid class widths."""
    with pytest.raises(ValueError, match="exactly 3 values"):
        ImageClassificationLoss(3, min_iou=[0.5, 0.5])
    with pytest.raises(ValueError, match="requires `num_classes`"):
        ImageRegressionLoss(1, min_iou=[0.5])

    objects = ImageObjectBuilder()(image_data)
    prediction = TensorBatch(torch.zeros((2, 3)), counts=objects.counts)
    with pytest.raises(TypeError, match="ClusterLabelBatch"):
        ImageClassificationLoss(3, min_iou=0.5)([0, 1], objects, prediction)

    regression = TensorBatch(torch.zeros((2, 1)), counts=objects.counts)
    with pytest.raises(TypeError, match="ClusterLabelBatch"):
        ImageRegressionLoss(1, min_iou=0.5)([0.0, 1.0], objects, regression)


def test_image_regression_supports_class_dependent_overlap_thresholds(image_data):
    """Regression gates may vary by a separate categorical truth field."""
    objects = ImageObjectBuilder(source="cluster")(
        image_data,
        object_data=image_data,
    )
    predictions = TensorBatch(torch.zeros((4, 1)), counts=objects.counts)
    thresholds = [0.0, 0.0, 0.75, 0.75, 0.0]

    result = ImageRegressionLoss(
        1,
        target="momentum",
        min_efficiency=thresholds,
        quality_num_classes=5,
    )(image_data, objects, predictions)

    assert result["count"] == 2


def test_image_overlap_quality_classes_must_be_scalar(image_data):
    """Vector-valued fields cannot index class-dependent thresholds."""
    objects = ImageObjectBuilder(source="cluster")(
        image_data,
        object_data=image_data,
    )
    predictions = TensorBatch(torch.zeros((4, 1)), counts=objects.counts)
    loss = ImageRegressionLoss(
        1,
        target="momentum",
        min_iou=[0.0, 0.0, 0.0],
        quality_target="vertex",
        quality_num_classes=3,
    )

    with pytest.raises(ValueError, match="scalar IDs"):
        loss(image_data, objects, predictions)


def test_ancestor_targets_use_root_particle_pid(image_data):
    """Ancestor PID must come from the root rather than the modal descendant."""
    ancestor_data = _with_shared_first_ancestor(image_data)
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
    ancestor_data = _with_shared_first_ancestor(image_data)
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


def test_invalid_ancestor_energy_is_ignored(image_data):
    """Unknown PID or unphysical momentum excludes an energy target."""
    ancestor_data = _with_shared_first_ancestor(image_data)
    ancestor_data.particles["pid"].data[0] = 999
    ancestor_data.particles["momentum"].data[1] = -1.0
    objects = ImageObjectBuilder(source="ancestor")(
        ancestor_data,
        object_data=ancestor_data,
    )
    predictions = TensorBatch(
        torch.zeros((len(objects.index_list), 1)),
        counts=objects.counts,
    )

    result = ImageRegressionLoss(
        1,
        target="kinetic_energy",
        target_reduction="ancestor",
    )(ancestor_data, objects, predictions)

    assert result["count"] < len(objects.index_list)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"out_channels": 1}, "at least two classes"),
        (
            {"out_channels": 2, "balance_loss": True, "class_weights": [1, 1]},
            "Cannot combine",
        ),
        ({"out_channels": 2, "class_weights": [1]}, "one weight per"),
        ({"out_channels": 2, "target_reduction": "bad"}, "mode.*ancestor"),
    ],
)
def test_classification_loss_validates_configuration(kwargs, message):
    """Classification topology, weighting, and reduction are checked early."""
    with pytest.raises(ValueError, match=message):
        ImageClassificationLoss(**kwargs)


@pytest.mark.parametrize(
    ("labels", "kwargs", "message"),
    [
        ("text", {}, "numeric values"),
        ([0], {}, "one value per image object"),
        (
            None,
            {"target": "pid"},
            "Voxel-level particle targets require ClusterLabelBatch",
        ),
        (
            None,
            {"target_reduction": "ancestor"},
            "Direct image labels do not support",
        ),
    ],
)
def test_classification_loss_validates_direct_targets(
    image_data,
    labels,
    kwargs,
    message,
):
    """Direct targets must be numeric, scalar, aligned, and unreduced."""
    objects = ImageObjectBuilder()(image_data)
    prediction = TensorBatch(torch.zeros((2, 3)), counts=[1, 1])
    if labels is None:
        labels = TensorBatch(torch.tensor([0, 1]), counts=[1, 1])
    with pytest.raises((TypeError, ValueError), match=message):
        ImageClassificationLoss(3, **kwargs)(labels, objects, prediction)


def test_structured_and_batched_targets_validate_required_shape(image_data):
    """Structured targets require names and direct batches match objects."""
    objects = ImageObjectBuilder()(image_data)
    prediction = TensorBatch(torch.zeros((2, 3)), counts=[1, 1])

    with pytest.raises(ValueError, match="named target"):
        ImageClassificationLoss(3)(image_data, objects, prediction)
    with pytest.raises(TypeError, match="must be named fields"):
        ImageClassificationLoss(3, target=0)(image_data, objects, prediction)
    with pytest.raises(ValueError, match="one row per image object"):
        ImageClassificationLoss(3)(
            TensorBatch(torch.tensor([0]), counts=[1]),
            objects,
            prediction,
        )

    column_targets = TensorBatch(torch.tensor([[0], [1]]), counts=[1, 1])
    result = ImageClassificationLoss(3)(column_targets, objects, prediction)
    assert result["count"] == 2


def test_classification_loss_covers_weighting_ignored_and_invalid_targets(image_data):
    """Classification handles dynamic/fixed weights and empty supervision."""
    objects = ImageObjectBuilder()(image_data)
    logits = TensorBatch(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), counts=[1, 1])

    balanced = ImageClassificationLoss(2, balance_loss=True)([0, 1], objects, logits)
    fixed = ImageClassificationLoss(2, class_weights=[1.0, 2.0])(
        [0, 1], objects, logits
    )
    ignored = ImageClassificationLoss(2)([-1, -1], objects, logits)
    assert balanced["accuracy"] == 1.0
    assert fixed["accuracy"] == 1.0
    assert ignored["count"] == 0
    assert ignored["accuracy_class_0"] == 1.0

    vector_labels = TensorBatch(torch.zeros((2, 2)), counts=[1, 1])
    with pytest.raises(ValueError, match="scalar class IDs"):
        ImageClassificationLoss(2)(vector_labels, objects, logits)
    with pytest.raises(ValueError, match="must lie"):
        ImageClassificationLoss(2)([0, 2], objects, logits)


def test_ancestor_reduction_rejects_unsupported_target(image_data):
    """Ancestor reduction is limited to fields with defined root semantics."""
    objects = ImageObjectBuilder(source="ancestor")(
        image_data,
        object_data=image_data,
    )
    logits = TensorBatch(torch.zeros((4, 2)), counts=objects.counts)
    with pytest.raises(ValueError, match="supports only `pid` and `kinetic_energy`"):
        ImageClassificationLoss(
            2,
            target="shape",
            target_reduction="ancestor",
        )(image_data, objects, logits)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"out_channels": 0}, "must be positive"),
        ({"out_channels": 1, "target_reduction": "bad"}, "mode.*ancestor"),
    ],
)
def test_regression_loss_validates_configuration(kwargs, message):
    """Regression output width and target reduction are constrained."""
    with pytest.raises(ValueError, match=message):
        ImageRegressionLoss(**kwargs)


def test_regression_loss_handles_shape_mismatch_and_empty_supervision(image_data):
    """Regression validates vector width and returns stable empty metrics."""
    objects = ImageObjectBuilder()(image_data)
    vector_prediction = TensorBatch(torch.zeros((2, 2)), counts=[1, 1])
    with pytest.raises(ValueError, match="shapes do not match"):
        ImageRegressionLoss(2)([1.0, 2.0], objects, vector_prediction)

    prediction = TensorBatch(torch.zeros((2, 1)), counts=[1, 1])
    result = ImageRegressionLoss(1)([-1.0, float("nan")], objects, prediction)
    assert result["bias"] == 0.0
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["count"] == 0
    assert result["loss"].item() == 0.0


@pytest.mark.parametrize(
    ("image", "loss", "message"),
    [
        ({}, {}, "requires model `heads`"),
        ({"heads": {"pid": []}}, {"pid": {}}, "determine output width"),
        ({"heads": {"pid": 2}}, {}, "exactly match"),
        ({"heads": {"pid": 2}}, {"pid": {}}, "requires `name`"),
        (
            {"heads": {"pid": 2}},
            {"pid": {"name": "class", "weight": 0}},
            "weights must be positive",
        ),
        (
            {"heads": {"pid": 2}},
            {"pid": {"name": "unknown"}},
            "Unknown image task",
        ),
    ],
)
def test_image_loss_validates_task_configuration(image, loss, message):
    """Loss tasks must map one-to-one onto valid model heads."""
    with pytest.raises(ValueError, match=message):
        ImageLoss(image, loss)


def test_image_loss_requires_prediction_and_label_inputs(image_data):
    """The orchestrator identifies missing model outputs and targets by name."""
    objects = ImageObjectBuilder()(image_data)
    loss = ImageLoss(
        {"heads": {"pid": {"out_channels": 2}}},
        {"pid": {"name": "class", "label": "labels"}},
    )
    with pytest.raises(ValueError, match="missing `pid_pred`"):
        loss(objects, labels=[0, 1])
    with pytest.raises(ValueError, match="missing `labels`"):
        loss(objects, pid_pred=TensorBatch(torch.zeros((2, 2)), counts=[1, 1]))
