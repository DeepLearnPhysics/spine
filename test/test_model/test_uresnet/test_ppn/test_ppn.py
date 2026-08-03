"""Focused tests for point-proposal helpers and configuration."""

import pytest
import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, TensorBatch, TensorSchema
from spine.model.uresnet.ppn import UResNetPPN
from spine.model.uresnet.ppn.ppn import PPN, ExpandAs, PPNLoss
from spine.model.uresnet.ppn.vertex import VertexPPNLoss
from spine.utils.ppn import ppn_raw_schema

POINT_SCHEMA = TensorSchema(coordinate_groups={"points": (0, 1, 2)})
PPN_LABEL_SCHEMA = TensorSchema(
    coordinate_groups={"point": (0, 1, 2)},
    feature_fields={"shape": (0,), "particle": (1,), "endpoint": (2,)},
)


class FeatureTensor:
    def __init__(self, features):
        self.features = features

    def replace_features(self, features):
        return FeatureTensor(features)


def test_expand_as_propagate_all_does_not_mutate_scores():
    features = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    tensor = FeatureTensor(features.clone())

    output = ExpandAs()(tensor, (2, 3), propagate_all=True)

    assert torch.equal(tensor.features, features)
    assert torch.equal(output.features, torch.ones((2, 3)))


def test_ppn_loss_normalizes_point_classes_to_tuple(cnn_config):
    loss = PPNLoss(cnn_config, {"point_classes": 2})

    assert loss.point_classes == (2,)


def test_ppn_takes_ghost_configuration_from_backbone(cnn_config):
    model = PPN({**cnn_config, "num_classes": 5, "ghost": True}, {})

    assert model.ghost
    assert model.masker is not None
    assert model.merge_concat is not None
    assert model.ghost_mask is not None


def test_ppn_rejects_true_ghost_mask_without_ghost_backbone(cnn_config):
    with pytest.raises(ValueError, match="UResNet `ghost: true`"):
        PPN(
            {**cnn_config, "num_classes": 5},
            {"use_true_ghost_mask": True},
        )


def test_ppn_rejects_duplicate_ghost_configuration(cnn_config):
    with pytest.raises(TypeError, match="ghost"):
        PPN(
            {**cnn_config, "num_classes": 5},
            {"ghost": True},
        )


def test_ppn_requires_multiple_resolution_levels(cnn_config):
    config = {**cnn_config, "num_classes": 5, "depth": 1}

    with pytest.raises(ValueError, match="depth of at least two"):
        PPN(config, {})
    with pytest.raises(ValueError, match="depth of at least two"):
        PPNLoss(config, {})


@pytest.mark.parametrize(
    ("loss_config", "message"),
    [
        ({"resolution": 0.0}, "resolution"),
        ({"mask_loss_weight": -1.0}, "mask_loss_weight"),
    ],
)
def test_ppn_loss_rejects_invalid_numeric_configuration(
    cnn_config,
    loss_config,
    message,
):
    with pytest.raises(ValueError, match=message):
        PPNLoss(cnn_config, loss_config)


def test_ppn_forwards_configured_threshold_to_binary_gating(cnn_config):
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5},
        {"mask_score_threshold": 0.8, "use_binary_mask": True},
    )
    model.eval()
    recorded_thresholds = []
    expand_as = model.ppn.expand_as

    class RecordingExpandAs(torch.nn.Module):
        def forward(self, *args, **kwargs):
            recorded_thresholds.append(kwargs["score_threshold"])
            return expand_as(*args, **kwargs)

    model.ppn.expand_as = RecordingExpandAs()
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )

    model(data)

    assert recorded_thresholds == [0.8] * (cnn_config["depth"] - 1)


def test_coordinate_alignment_handles_reordering_and_duplicates():
    source_coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
        ]
    )
    source_values = torch.tensor([7, 7, 9])
    target_coords = torch.tensor([[0, 2, 2, 2], [0, 1, 1, 1]])

    aligned = PPNLoss.align_coordinate_values(
        source_coords,
        source_values,
        target_coords,
        "particle label",
    )

    assert torch.equal(aligned, torch.tensor([9, 7]))


def test_coordinate_alignment_rejects_conflicting_duplicates():
    coords = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])

    with pytest.raises(ValueError, match="conflicting"):
        PPNLoss.align_coordinate_values(
            coords,
            torch.tensor([7, 8]),
            coords[:1],
            "particle label",
        )


def test_coordinate_alignment_can_fill_missing_mask_sites():
    """Generated coarse sparse sites without fine children are negatives."""
    source_coords = torch.tensor([[0, 1, 1, 1]])
    target_coords = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]])

    aligned = PPNLoss.align_coordinate_values(
        source_coords,
        torch.tensor([1.0]),
        target_coords,
        "PPN mask",
        missing_value=0,
    )

    assert torch.equal(aligned, torch.tensor([1.0, 0.0]))


def test_ppn_loss_handles_an_empty_batch(cnn_config):
    loss = PPNLoss(cnn_config, {})
    counts = [0, 0]
    coords = TensorBatch(
        torch.empty((0, 4)),
        counts=counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    scores = TensorBatch(torch.empty((0, 2)), counts=counts)
    masks = TensorBatch(torch.empty((0, 1), dtype=torch.bool), counts=counts)
    points = TensorBatch(torch.empty((0, 10)), counts=counts, schema=ppn_raw_schema())
    labels = TensorBatch(
        torch.empty((0, 7)),
        counts=counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=PPN_LABEL_SCHEMA,
    )

    result = loss(
        ppn_label=labels,
        ppn_points=points,
        ppn_points_unique=points,
        ppn_masks=[masks],
        ppn_layers=[scores],
        ppn_coords=[coords],
        ppn_output_coords=coords,
    )

    assert torch.isfinite(result["loss"])


def test_cluster_restriction_aligns_duplicate_input_rows(cnn_config):
    loss = PPNLoss(cnn_config, {"restrict_to_clusters": True})
    coords = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    scores = TensorBatch(torch.tensor([[0.0, 1.0]]), counts=[1])
    masks = TensorBatch(torch.tensor([[True]]), counts=[1])
    points = TensorBatch(torch.zeros((1, 10)), counts=[1], schema=ppn_raw_schema())
    ppn_tensor = torch.zeros((1, 7))
    ppn_tensor[0, 1:4] = 1.5
    ppn_tensor[0, 4] = 0
    ppn_tensor[0, 5] = 11
    ppn_label = TensorBatch(
        ppn_tensor,
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=PPN_LABEL_SCHEMA,
    )
    clust_tensor = torch.zeros((2, 7))
    clust_tensor[:, :4] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    clust_tensor[:, 5] = 0
    clust_tensor[:, 6] = 0
    clust_label = ClusterLabelBatch(
        TensorBatch(
            clust_tensor,
            counts=[2],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        ),
        {"particle": TensorBatch(torch.tensor([11]), counts=[1])},
    )

    result = loss(
        ppn_label=ppn_label,
        clust_label=clust_label,
        ppn_points=points,
        ppn_points_unique=points,
        ppn_masks=[masks],
        ppn_layers=[scores],
        ppn_coords=[coords],
        ppn_output_coords=coords,
    )

    assert torch.isfinite(result["loss"])


def test_true_ghost_mask_prunes_propagated_and_skip_features(cnn_config):
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5, "ghost": True},
        {"use_true_ghost_mask": True},
    )
    model.eval()
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 6.0, 6.0, 6.0, 3.0],
            ]
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    seg_label = TensorBatch(torch.tensor([0.0, float(GHOST_SHP)]), counts=[2])

    result = model(data, seg_label)

    final_coords = result["ppn_coords"][-1].torch_tensor()
    assert final_coords.shape[0] == 1
    assert torch.equal(final_coords[0, 1:], torch.tensor([1, 1, 1]))


def test_vertex_loss_reports_unsupported_label_contract():
    loss = VertexPPNLoss()

    with pytest.raises(NotImplementedError, match="vertex-label schema"):
        loss()
