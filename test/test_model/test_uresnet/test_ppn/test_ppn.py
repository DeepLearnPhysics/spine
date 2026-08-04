"""Focused tests for point-proposal helpers and configuration."""

import pytest
import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, TensorBatch, TensorSchema
from spine.model import sparse
from spine.model.cnn.uresnet_layers import UResNet
from spine.model.uresnet.ppn import UResNetPPN
from spine.model.uresnet.ppn.ppn import (
    PPN,
    AttentionMask,
    ExpandAs,
    GhostMask,
    MergeConcat,
    PPNLoss,
)
from spine.model.uresnet.ppn.vertex import VertexPPN, VertexPPNLoss
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


def test_ppn_validates_decoder_feature_count(cnn_config):
    """The proposal decoder requires one UResNet feature map per level."""
    model = PPN({**cnn_config, "num_classes": 5}, {})
    with pytest.raises(ValueError, match="decoder tensors"):
        model(object(), [])


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


def test_ghost_ppn_validates_and_uses_ghost_inputs(cnn_config):
    """Ghost PPN requires its selected mask source and supports predictions."""
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    truth_model = UResNetPPN(
        {**cnn_config, "num_classes": 5, "ghost": True},
        {"use_true_ghost_mask": True},
    )
    truth_model.eval()
    with pytest.raises(ValueError, match="provide the `seg_label`"):
        truth_model(data)

    backbone_result = truth_model.uresnet(data)
    with pytest.raises(ValueError, match="provide the `seg_label`"):
        truth_model.ppn(
            backbone_result["final_tensor"],
            backbone_result["decoder_tensors"],
            backbone_result["ghost_tensor"],
        )
    with pytest.raises(ValueError, match="label tensor length"):
        truth_model.ppn(
            backbone_result["final_tensor"],
            backbone_result["decoder_tensors"],
            backbone_result["ghost_tensor"],
            TensorBatch(torch.zeros(2), counts=[2]),
        )

    predicted_model = UResNetPPN(
        {**cnn_config, "num_classes": 5, "ghost": True},
        {},
    )
    predicted_model.eval()
    result = predicted_model(data)
    assert result["ppn_points"].shape[1] == 10

    backbone_result = predicted_model.uresnet(data)
    with pytest.raises(ValueError, match="ghost prediction logits"):
        predicted_model.ppn(
            backbone_result["final_tensor"],
            backbone_result["decoder_tensors"],
        )


def test_vertex_loss_reports_unsupported_label_contract():
    loss = VertexPPNLoss()

    with pytest.raises(NotImplementedError, match="vertex-label schema"):
        loss()


def test_vertex_ppn_decodes_uresnet_feature_pyramid(cnn_config):
    """Vertex prediction consumes the maintained UResNet pyramid contract."""
    rows = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                rows.append([0.0, x, y, z, x + y + z + 1.0])
    table = torch.tensor(rows, dtype=torch.float32)
    backbone = UResNet(cnn_config)
    vertex = VertexPPN(cnn_config, {"score_threshold": 0.7})

    features = backbone(table)
    result = vertex(features["final_tensor"], features["decoder_tensors"])

    assert result["vertex_points"].shape == (len(table), 5)
    assert result["vertex_points_unique"].shape[1] == 5
    assert len(result["vertex_layers"]) == cnn_config["depth"] - 1
    assert len(result["vertex_coords"]) == cnn_config["depth"] - 1
    assert result["vertex_output_coordinates"].has_batch_col


def test_vertex_ppn_validates_threshold_and_feature_pyramid(cnn_config):
    """Vertex configuration and decoder depth fail before ambiguous execution."""
    with pytest.raises(ValueError, match="between zero and one"):
        VertexPPN(cnn_config, {"score_threshold": 1.1})

    vertex = VertexPPN(cnn_config)
    with pytest.raises(ValueError, match="decoder tensors"):
        vertex(object(), [])


def test_ppn_validates_required_and_optional_configuration(cnn_config):
    """PPN reports malformed model and loss configuration immediately."""
    with pytest.raises(ValueError, match="num_classes"):
        PPN(cnn_config, {})
    with pytest.raises(ValueError, match="between zero and one"):
        PPN({**cnn_config, "num_classes": 5}, {"mask_score_threshold": 1.1})
    with pytest.raises(ValueError, match="define `depth`"):
        PPNLoss({}, {})
    with pytest.raises(ValueError, match="not recognized"):
        PPNLoss(cnn_config, {"mask_loss": "dice"})

    loss = PPNLoss(cnn_config, {"point_classes": [1, 2]})
    assert loss.point_classes == (1, 2)


def test_coordinate_alignment_validates_shapes_and_missing_sites():
    """Coordinate alignment rejects inconsistent source and target tables."""
    coords = torch.tensor([[0, 1, 1, 1]])
    with pytest.raises(ValueError, match="matching lengths"):
        PPNLoss.align_coordinate_values(
            coords,
            torch.tensor([1, 2]),
            coords,
            "label",
        )
    with pytest.raises(ValueError, match="same width"):
        PPNLoss.align_coordinate_values(
            coords,
            torch.tensor([1]),
            torch.tensor([[0, 1, 1]]),
            "label",
        )
    with pytest.raises(ValueError, match="missing"):
        PPNLoss.align_coordinate_values(
            coords,
            torch.tensor([1]),
            torch.tensor([[0, 2, 2, 2]]),
            "label",
        )


def test_get_ppn_positives_validates_restricted_inputs(monkeypatch):
    """Positive construction requires distances and particle point labels."""
    coords = torch.zeros((1, 3))
    points = torch.zeros((1, 3))
    monkeypatch.setattr(
        "spine.model.uresnet.ppn.ppn.cdist_fast",
        lambda *args: None,
    )
    with pytest.raises(RuntimeError, match="distance matrix"):
        PPNLoss.get_ppn_positives(coords, points, 1.0, 0)

    monkeypatch.setattr(
        "spine.model.uresnet.ppn.ppn.cdist_fast",
        lambda *args: torch.zeros((1, 1)),
    )
    with pytest.raises(ValueError, match="particle labels"):
        PPNLoss.get_ppn_positives(
            coords,
            points,
            1.0,
            0,
            labels=torch.zeros(1),
        )


def _positive_ppn_loss_inputs():
    """Build a one-site, one-track target PPN loss input."""
    coords = TensorBatch(
        torch.tensor([[0, 0, 0, 0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    scores = TensorBatch(torch.tensor([[0.0, 2.0]]), counts=[1])
    masks = TensorBatch(torch.tensor([[True]]), counts=[1])

    point_features = torch.zeros((1, 10))
    point_features[0, 4] = 3.0
    points = TensorBatch(point_features, counts=[1], schema=ppn_raw_schema())

    label = TensorBatch(
        torch.tensor([[0.0, 0.5, 0.5, 0.5, 1.0, 7.0, 1.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=PPN_LABEL_SCHEMA,
    )
    endpoint_schema = TensorSchema(
        feature_fields={"endpoint_logits": (0, 1)},
        feats_only=True,
    )
    endpoints = TensorBatch(
        torch.tensor([[0.0, 2.0]]),
        counts=[1],
        schema=endpoint_schema,
    )

    return {
        "ppn_label": label,
        "ppn_points": points,
        "ppn_points_unique": points,
        "ppn_masks": [masks],
        "ppn_layers": [scores],
        "ppn_coords": [coords],
        "ppn_output_coords": coords,
        "ppn_classify_endpoints": endpoints,
        "ppn_classify_endpoints_unique": endpoints,
    }


def test_ppn_loss_supervises_endpoint_and_returns_masks(cnn_config):
    """A positive track site trains all PPN heads and exposes mask labels."""
    result = PPNLoss(
        cnn_config,
        {
            "balance_mask_loss": False,
            "balance_type_loss": False,
            "return_mask_labels": True,
        },
    )(**_positive_ppn_loss_inputs())

    assert torch.isfinite(result["loss"])
    assert result["type_accuracy"] == 1.0
    assert result["classify_endpoints_accuracy"] == 1.0
    assert len(result["mask_labels"]) == 1


def test_ppn_loss_filters_requested_point_classes(cnn_config):
    """A nonempty point-class filter retains matching proposal labels."""
    result = PPNLoss(cnn_config, {"point_classes": [1]})(**_positive_ppn_loss_inputs())
    assert torch.isfinite(result["loss"])


def test_ppn_loss_validates_output_alignment(cnn_config):
    """PPN loss refuses incomplete pyramids and row-misaligned heads."""
    inputs = _positive_ppn_loss_inputs()
    loss = PPNLoss(cnn_config, {})

    with pytest.raises(ValueError, match="Expected 1 PPN layers"):
        loss(**{**inputs, "ppn_masks": []})

    empty_class_loss = PPNLoss(cnn_config, {"point_classes": []})
    with pytest.raises(ValueError, match="at least one class"):
        empty_class_loss(**inputs)

    other_coords = TensorBatch(
        torch.tensor([[0, 1, 1, 1]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    with pytest.raises(ValueError, match="must match"):
        loss(**{**inputs, "ppn_output_coords": other_coords})

    extra_points = TensorBatch(
        torch.zeros((2, 10)),
        counts=[2],
        schema=ppn_raw_schema(),
    )
    with pytest.raises(ValueError, match="point predictions"):
        loss(**{**inputs, "ppn_points_unique": extra_points})

    extra_endpoints = TensorBatch(torch.zeros((2, 2)), counts=[2])
    with pytest.raises(ValueError, match="endpoint predictions"):
        loss(
            **{
                **inputs,
                "ppn_classify_endpoints_unique": extra_endpoints,
            }
        )

    with pytest.raises(ValueError, match="clust_label"):
        PPNLoss(cnn_config, {"restrict_to_clusters": True})(**inputs)

    extra_masks = TensorBatch(torch.ones((2, 1), dtype=torch.bool), counts=[2])
    with pytest.raises(ValueError, match="mask and score rows"):
        loss(**{**inputs, "ppn_masks": [extra_masks]})


def test_ppn_endpoint_head_runs_with_backbone(cnn_config):
    """Endpoint-enabled PPN returns aligned original and unique logits."""
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5},
        {"classify_endpoints": True},
    )
    model.eval()
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )

    result = model(data)

    assert result["ppn_classify_endpoints"].shape == (1, 2)
    assert result["ppn_classify_endpoints_unique"].shape == (1, 2)


def test_sparse_ppn_helpers_validate_and_transform():
    """Mask helpers validate channels, thresholds, dimensions and strides."""
    feature = FeatureTensor(torch.ones((1, 3)))
    with pytest.raises(ValueError, match="two-score"):
        ExpandAs()(feature, (1, 3))
    with pytest.raises(ValueError, match="between zero and one"):
        AttentionMask(1.1)
    with pytest.raises(ValueError, match="positive"):
        GhostMask(0)

    expanded = ExpandAs()(
        FeatureTensor(torch.tensor([[0.9, 0.1]])),
        (1, 2),
        use_binary_mask=True,
        score_threshold=0.5,
    )
    assert not torch.any(expanded.features)

    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    first = sparse.SparseTensor(torch.ones((1, 1)), coordinates=coords)
    second = sparse.SparseTensor(
        torch.ones((1, 1)),
        coordinates=coords,
        tensor_stride=2,
    )
    with pytest.raises(ValueError, match="tensor_stride"):
        AttentionMask()(first, second)
    with pytest.raises(ValueError, match="tensor_stride"):
        MergeConcat()(first, second)

    target = sparse.SparseTensor(
        torch.ones((1, 1)),
        coordinates=coords,
        tensor_stride=3,
    )
    with pytest.raises(ValueError, match="power-of-two"):
        GhostMask()(first, target)


def test_sparse_ppn_helpers_union_coordinates_and_downsample_masks():
    """Sparse merge and mask helpers execute their nontrivial resolution paths."""
    first_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    other_coords = torch.tensor([[0, 1, 0, 0]], dtype=torch.int32)
    first = sparse.SparseTensor(torch.ones((1, 1)), coordinates=first_coords)
    other = sparse.SparseTensor(
        torch.full((1, 1), 2.0),
        coordinates=other_coords,
        coordinate_manager=first.coordinate_manager,
    )
    merged = MergeConcat()(first, other)
    assert merged.features.shape == (2, 2)

    ghost_coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    ghost = sparse.SparseTensor(torch.tensor([[0.0], [1.0]]), coordinates=ghost_coords)
    target = sparse.SparseTensor(
        torch.ones((1, 1)), coordinates=first_coords, tensor_stride=2
    )
    downsampled = GhostMask()(ghost, target)
    assert downsampled.tensor_stride[0] == 2
