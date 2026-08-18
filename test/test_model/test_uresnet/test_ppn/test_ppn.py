"""Focused tests for point-proposal helpers and configuration."""

import pytest
import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, TensorBatch, TensorSchema
from spine.model import sparse
from spine.model.cnn.uresnet_layers import UResNet
from spine.model.uresnet.ppn import (
    PointProposalDecoder,
    ProposalTask,
    UResNetPPN,
    UResNetPPNLoss,
)
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
VERTEX_LABEL_SCHEMA = TensorSchema(
    coordinate_groups={"vertex": (0, 1, 2)},
    feature_fields={"interaction": (0,)},
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


def _positive_vertex_loss_inputs():
    """Build one final-resolution site centered on its target vertex."""
    coords = TensorBatch(
        torch.tensor([[0, 0, 0, 0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    logits = TensorBatch(torch.tensor([[0.0, 2.0]]), counts=[1])
    masks = TensorBatch(torch.tensor([[True]]), counts=[1])
    points = TensorBatch(
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0]]),
        counts=[1],
        schema=TensorSchema(
            feature_fields={
                "offsets": (0, 1, 2),
                "vertex_logits": (3, 4),
            },
            feats_only=True,
        ),
    )
    label = TensorBatch(
        torch.tensor([[0.0, 0.5, 0.5, 0.5, 7.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=VERTEX_LABEL_SCHEMA,
    )
    return {
        "vertex_label": label,
        "vertex_points": points,
        "vertex_points_unique": points,
        "vertex_masks": [masks],
        "vertex_layers": [logits],
        "vertex_coords": [coords],
        "vertex_output_coords": coords,
    }


def test_vertex_loss_trains_foreground_and_offsets(cnn_config):
    """The parser-style vertex contract supervises both proposal outputs."""
    loss = VertexPPNLoss(
        cnn_config,
        {
            "balance_mask_loss": False,
            "return_mask_labels": True,
        },
    )

    result = loss(**_positive_vertex_loss_inputs())

    assert torch.isfinite(result["loss"])
    assert result["mask_accuracy"] == 1.0
    assert result["reg_accuracy"] == 1.0
    assert len(result["mask_labels"]) == 1


def test_vertex_loss_handles_empty_labels_and_predictions(cnn_config):
    """Empty entries produce a finite differentiable zero objective."""
    counts = [0, 0]
    coords = TensorBatch(
        torch.empty((0, 4)),
        counts=counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    logits = TensorBatch(torch.empty((0, 2)), counts=counts)
    masks = TensorBatch(torch.empty((0, 1), dtype=torch.bool), counts=counts)
    points = TensorBatch(
        torch.empty((0, 5)),
        counts=counts,
        schema=TensorSchema(
            feature_fields={
                "offsets": (0, 1, 2),
                "vertex_logits": (3, 4),
            },
            feats_only=True,
        ),
    )
    labels = TensorBatch(
        torch.empty((0, 5)),
        counts=counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=VERTEX_LABEL_SCHEMA,
    )

    result = VertexPPNLoss(cnn_config, {})(
        vertex_label=labels,
        vertex_points=points,
        vertex_masks=[masks],
        vertex_layers=[logits],
        vertex_coords=[coords],
        vertex_output_coords=coords,
    )

    assert torch.isfinite(result["loss"])


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
    assert result["vertex_output_coords"].has_batch_col


@pytest.mark.parametrize("shared", [False, True])
def test_combined_point_tasks_support_decoder_sharing(cnn_config, shared):
    """Dual-task models expose the same products in both decoder modes."""
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5},
        ppn={},
        vertex={},
        proposal_decoder={"shared": shared},
    )
    model.eval()
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )

    result = model(data)

    assert "ppn_points" in result
    assert "vertex_points" in result
    assert (model.vertex is None) == shared
    assert hasattr(model.ppn, "vertex_pred") == shared


def test_vertex_only_model_uses_the_generic_proposal_decoder(cnn_config):
    """Vertex regression can run without constructing particle-point heads."""
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5},
        vertex={},
    )
    model.eval()
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )

    result = model(data)

    assert "vertex_points" in result
    assert "ppn_points" not in result
    assert model.ppn is None
    assert model.vertex is not None


def test_vertex_ppn_validates_threshold_and_feature_pyramid(cnn_config):
    """Vertex configuration and decoder depth fail before ambiguous execution."""
    with pytest.raises(ValueError, match="between zero and one"):
        VertexPPN(cnn_config, {"score_threshold": 1.1})

    vertex = VertexPPN(cnn_config)
    with pytest.raises(ValueError, match="decoder tensors"):
        vertex(object(), [])


def test_proposal_task_configuration_is_validated(cnn_config):
    """Cross-task configuration fails early when its intent is ambiguous."""
    backbone = {**cnn_config, "num_classes": 5}
    with pytest.raises(ValueError, match="at least one"):
        UResNetPPN(backbone)
    with pytest.raises(ValueError, match="requires both"):
        UResNetPPN(backbone, ppn={}, proposal_decoder={"shared": True})
    with pytest.raises(TypeError, match="Unexpected proposal-decoder"):
        UResNetPPN(backbone, ppn={}, proposal_decoder={"mode": "shared"})

    with pytest.raises(ValueError, match="requires `vertex_loss`"):
        UResNetPPNLoss(
            uresnet=backbone,
            uresnet_loss={},
            vertex={},
        )

    with pytest.raises(ValueError, match="at least one proposal task loss"):
        UResNetPPNLoss(backbone, {})
    with pytest.raises(ValueError, match="requires `ppn_loss`"):
        UResNetPPNLoss(backbone, {}, ppn={})


def test_generic_proposal_decoder_validates_task_identity(cnn_config):
    """Generic proposal paths require unique task and module names."""

    class RawPointProposalDecoder(PointProposalDecoder):
        """Expose raw decoder products for generic contract tests."""

        def forward(self, *args, **kwargs):
            return self.decode(*args, **kwargs)

    with pytest.raises(ValueError, match="At least one"):
        RawPointProposalDecoder(cnn_config, [])
    with pytest.raises(ValueError, match="task names"):
        RawPointProposalDecoder(
            cnn_config,
            [ProposalTask("point", "first", 0.5), ProposalTask("point", "second", 0.5)],
        )
    with pytest.raises(ValueError, match="module names"):
        RawPointProposalDecoder(
            cnn_config,
            [
                ProposalTask("first", "scores", 0.5),
                ProposalTask("second", "scores", 0.5),
            ],
        )

    # The generic form is directly executable for consumers that only need
    # the final feature plane and multiscale proposal products.
    rows = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    features = UResNet(cnn_config)(rows)
    decoder = RawPointProposalDecoder(
        cnn_config,
        [ProposalTask("custom", "custom_masks", 0.5)],
        legacy_layers=False,
    )
    decoder.eval()
    final, outputs = decoder(
        features["final_tensor"],
        features["decoder_tensors"],
    )
    assert final.shape[0] == 1
    assert len(outputs["custom"]["layers"]) == 1


@pytest.mark.parametrize("gate_option", ["propagate_all", "use_binary_mask"])
def test_shared_decoder_supports_configured_union_gating(cnn_config, gate_option):
    """Shared paths combine task scores under soft and configured hard gates."""
    model = UResNetPPN(
        {**cnn_config, "num_classes": 5},
        ppn={gate_option: True},
        vertex={},
        proposal_decoder={"shared": True},
    )
    model.eval()
    data = TensorBatch(
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )

    result = model(data)

    assert result["ppn_points"].shape[0] == 1
    assert result["vertex_points"].shape[0] == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"extra": True}, "Unexpected vertex configuration"),
        ({"mask_score_threshold": -0.1}, "between zero and one"),
    ],
)
def test_shared_vertex_configuration_is_validated(cnn_config, config, message):
    """Shared vertex heads apply the same strict contract as standalone ones."""
    with pytest.raises((TypeError, ValueError), match=message):
        PPN({**cnn_config, "num_classes": 5}, {}, vertex=config)


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"extra": True}, TypeError, "Unexpected vertex-loss"),
        ({"mask_loss": "dice"}, ValueError, "not recognized"),
        ({"resolution": 0.0}, ValueError, "resolution"),
        ({"reg_loss_weight": -1.0}, ValueError, "nonnegative"),
    ],
)
def test_vertex_loss_configuration_is_validated(
    cnn_config,
    config,
    error,
    message,
):
    """Malformed vertex objectives fail during construction."""
    with pytest.raises(error, match=message):
        VertexPPNLoss(cnn_config, config)

    if config == {"extra": True}:
        with pytest.raises(TypeError, match="Unexpected vertex configuration"):
            VertexPPN(cnn_config, config)


def test_vertex_loss_requires_a_multiscale_backbone(cnn_config):
    """Vertex supervision requires explicit depth and at least two scales."""
    with pytest.raises(ValueError, match="define `depth`"):
        VertexPPNLoss({}, {})
    with pytest.raises(ValueError, match="depth of at least two"):
        VertexPPNLoss({**cnn_config, "depth": 1}, {})


def test_vertex_loss_validates_prediction_alignment(cnn_config):
    """Vertex supervision rejects incomplete or row-misaligned products."""
    inputs = _positive_vertex_loss_inputs()
    loss = VertexPPNLoss(cnn_config, {})
    with pytest.raises(ValueError, match="Expected 1 vertex layers"):
        loss(**{**inputs, "vertex_masks": []})

    other_coords = TensorBatch(
        torch.tensor([[0, 1, 1, 1]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=POINT_SCHEMA,
    )
    with pytest.raises(ValueError, match="must match"):
        loss(**{**inputs, "vertex_output_coords": other_coords})

    extra_points = TensorBatch(
        torch.zeros((2, 5)),
        counts=[2],
        schema=inputs["vertex_points"].schema,
    )
    with pytest.raises(ValueError, match="matching rows"):
        loss(**{**inputs, "vertex_points_unique": extra_points})

    extra_masks = TensorBatch(torch.ones((2, 1), dtype=torch.bool), counts=[2])
    with pytest.raises(ValueError, match="mask and score rows"):
        loss(**{**inputs, "vertex_masks": [extra_masks]})


def test_combined_loss_routes_configured_proposal_tasks(cnn_config):
    """The wrapper requires each label and prefixes every native metric."""
    loss = UResNetPPNLoss(
        {**cnn_config, "num_classes": 5},
        uresnet_loss={},
        ppn={},
        ppn_loss={},
        vertex={},
        vertex_loss={},
        proposal_decoder={"shared": True},
    )

    class FixedLoss(torch.nn.Module):
        """Return a deterministic differentiable metric dictionary."""

        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, *args, **kwargs):
            del args, kwargs
            return {
                "loss": torch.tensor(float(self.value)),
                "accuracy": float(self.value),
            }

    loss.seg_loss = FixedLoss(1)
    loss.ppn_loss = FixedLoss(2)
    loss.vertex_loss = FixedLoss(3)
    label = TensorBatch(torch.zeros(1), counts=[1])

    with pytest.raises(ValueError, match="requires `ppn_label`"):
        loss(seg_label=label)
    with pytest.raises(ValueError, match="requires `vertex_label`"):
        loss(seg_label=label, ppn_label=label)

    result = loss(
        seg_label=label,
        ppn_label=label,
        vertex_label=label,
    )

    assert result["loss"] == 6.0
    assert result["accuracy"] == 2.0
    assert result["uresnet_loss"] == 1.0
    assert result["ppn_loss"] == 2.0
    assert result["vertex_loss"] == 3.0


def test_vertex_only_loss_constructs_without_ppn(cnn_config):
    """Vertex-only training does not require a particle-point loss block."""
    loss = UResNetPPNLoss(
        {**cnn_config, "num_classes": 5},
        uresnet_loss={},
        vertex={},
        vertex_loss={},
    )

    assert loss.ppn_loss is None
    assert isinstance(loss.vertex_loss, VertexPPNLoss)


def test_vertex_loss_builds_multiscale_targets(cnn_config):
    """A deeper decoder max-pools final labels onto every proposal scale."""
    config = {**cnn_config, "depth": 3}
    rows = [[0.0, x, y, z, 1.0] for x in range(4) for y in range(4) for z in range(4)]
    backbone = UResNet(config)
    vertex = VertexPPN(config)
    backbone.eval()
    vertex.eval()
    features = backbone(torch.tensor(rows))
    result = vertex(features["final_tensor"], features["decoder_tensors"])
    label = TensorBatch(
        torch.tensor([[0.0, 1.5, 1.5, 1.5, 0.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=VERTEX_LABEL_SCHEMA,
    )

    metrics = VertexPPNLoss(config, {})(vertex_label=label, **result)

    assert torch.isfinite(metrics["loss"])
    assert "mask_loss_layer_1" in metrics


def test_ppn_loss_builds_multiscale_targets(cnn_config):
    """Particle-point PPN retains target pooling on deeper generic decoders."""
    config = {**cnn_config, "num_classes": 5, "depth": 3}
    rows = [[0.0, x, y, z, 1.0] for x in range(4) for y in range(4) for z in range(4)]
    model = UResNetPPN(config, ppn={})
    model.eval()
    result = model(
        TensorBatch(
            torch.tensor(rows),
            counts=[len(rows)],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        )
    )
    label = TensorBatch(
        torch.tensor([[0.0, 1.5, 1.5, 1.5, 1.0, 0.0, 0.0]]),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=PPN_LABEL_SCHEMA,
    )

    metrics = PPNLoss(config, {})(ppn_label=label, **result)

    assert torch.isfinite(metrics["loss"])
    assert "mask_loss_layer_1" in metrics


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
