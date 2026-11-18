"""Shared-backbone execution, training and final-fragment contracts."""

import pytest
import torch

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch, TensorSchema
from spine.model.cnn.uresnet_layers import UResNet
from spine.model.full_chain import FullChain, FullChainLoss
from spine.model.full_chain.providers.multi import MultiLossStage
from spine.model.graph_spice import GraphSPICEEmbeddingHead, GraphSPICEHead
from spine.model.uresnet.multi import UResNetMulti, UResNetMultiLoss


@pytest.fixture
def config():
    return {
        "uresnet": {
            "num_classes": 5,
            "num_input": 1,
            "filters": 4,
            "depth": 2,
            "reps": 1,
            "spatial_size": 4,
            "activation": "relu",
            "norm_layer": "none",
        },
        "ppn": {},
        "graph_spice": {
            "use_raw_features": True,
            "embedder": {},
            "kernel": {"provider": "bilinear", "num_features": 4},
            "constructor": {
                "graph": {"provider": "radius", "r": 1.9},
                "min_size": 1,
                "edge_threshold": 0.5,
            },
        },
    }


def inputs():
    rows = torch.tensor(
        [
            [float(b), x, y, z, 1.0 + x]
            for b in (0, 2)
            for x in range(4)
            for y in range(4)
            for z in range(4)
        ]
    )
    data = TensorBatch(rows, [64, 0, 64], has_batch_col=True, coord_cols=(1, 2, 3))
    semantic = (rows[:, 1] >= 2).long()
    seg = TensorBatch(semantic, data.counts)
    clusters = ClusterLabelBatch(
        TensorBatch(
            torch.column_stack((rows, semantic.float())),
            data.counts,
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        )
    )
    ppn = TensorBatch(
        torch.tensor(
            [[0.0, 1.5, 1.5, 1.5, 1.0, 0.0, 0.0], [2.0, 1.5, 1.5, 1.5, 1.0, 0.0, 0.0]]
        ),
        [1, 0, 1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
        schema=TensorSchema(
            coordinate_groups={"point": (0, 1, 2)},
            feature_fields={"shape": (0,), "particle": (1,), "endpoint": (2,)},
        ),
    )
    return data, seg, clusters, ppn


def force_routing(model, classes):
    """Make routing deterministic without disconnecting semantic gradients."""

    def route(_module, _args, result):
        logits = result["segmentation"].torch_tensor()
        logits = logits * 0.01 + 10 * torch.nn.functional.one_hot(classes, 5)
        result["segmentation"] = TensorBatch(logits, result["segmentation"].counts)

    return model.uresnet_ppn.uresnet.register_forward_hook(route)


def loss_config():
    return {
        "uresnet_loss": {},
        "ppn_loss": {},
        "graph_spice_loss": {"provider": "edge", "metric": None},
    }


def test_one_backbone_and_coherent_fragments_with_filtered_rows(config):
    model = UResNetMulti(**config).eval()
    assert sum(isinstance(module, UResNet) for module in model.modules()) == 1
    assert not any("backbone" in key for key in model.graph_spice.state_dict())
    data, seg, _, _ = inputs()
    classes = seg.torch_tensor().clone()
    classes[::7] = 4  # Excluded voxels exercise restoration of filtered indexes.
    force_routing(model, classes)
    calls = []
    model.uresnet_ppn.uresnet.backbone.register_forward_hook(
        lambda *_args: calls.append(1)
    )
    result = model(data)
    assert calls == [1]
    assert result["fragment_clusts"].counts[1] == 0
    assert result["ppn_points"].shape[0] == data.shape[0]
    assert len(result["fragment_clusts"].index_list) > 0
    for cluster, shape in zip(
        result["fragment_clusts"].index_list, result["fragment_shapes"].tensor
    ):
        assert torch.all(result["seg_pred"].tensor[cluster] == int(shape))
        assert len(torch.unique(data.tensor[cluster, 0])) == 1
    assert torch.equal(result["seg_pred_raw"].tensor, classes)
    assert not any(4 == int(s) for s in result["fragment_shapes"].tensor)


@pytest.mark.parametrize("policy", ["sum", "fixed", "uncertainty"])
def test_joint_loss_backpropagates_all_tasks(config, policy):
    config["graph_spice"]["make_clusters"] = False
    model = UResNetMulti(**config).eval()
    data, seg, clusters, ppn = inputs()
    force_routing(model, seg.tensor)
    result = model(data)
    objective = UResNetMultiLoss(
        **config, **loss_config(), loss_balancing={"provider": policy}
    )
    metrics = objective(seg, clusters, ppn, **result)
    assert torch.isfinite(metrics["loss"])
    backbone_parameter = next(model.uresnet_ppn.uresnet.backbone.parameters())
    for name in ("uresnet_loss", "ppn_loss", "graph_spice_loss"):
        gradient = torch.autograd.grad(
            metrics[name], backbone_parameter, retain_graph=True
        )[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    metrics["loss"].backward()
    assert any(p.grad is not None for p in model.graph_spice.kernel_fn.parameters())
    if policy == "sum":
        torch.testing.assert_close(
            metrics["loss"],
            sum(metrics[k] for k in ("uresnet_loss", "ppn_loss", "graph_spice_loss")),
        )


def test_shared_head_preserves_alignment_and_handles_empty_selection(config):
    data, seg, _, _ = inputs()
    features = TensorBatch(
        torch.randn(data.shape[0], 4, requires_grad=True), data.counts
    )
    head = GraphSPICEHead(config["graph_spice"], 4, 4)
    result = head(data, seg, features)
    result["edge_attr"].tensor.sum().backward()
    assert features.tensor.grad is not None
    excluded = TensorBatch(torch.full((data.shape[0],), 4), data.counts)
    result = head(data, excluded, features)
    assert result["edge_attr"].shape[0] == 0
    head.zero_grad(set_to_none=True)
    result["edge_attr"].tensor.sum().backward()
    assert all(p.grad is not None for p in head.kernel_fn.parameters())
    with pytest.raises(ValueError, match="batch counts"):
        head(data, seg, TensorBatch(features.tensor, [0, 64, 64]))


def test_projected_embedding_head_has_no_backbone():
    data, _, _, _ = inputs()
    features = TensorBatch(torch.randn(data.shape[0], 4), data.counts)
    head = GraphSPICEEmbeddingHead(4, 4, feature_embedding_dim=2)
    result = head(data, features)
    torch.testing.assert_close(
        result["hypergraph_features"].tensor[:, :3], result["spatial_embeddings"].tensor
    )
    assert not hasattr(head, "backbone")


@pytest.mark.parametrize("mode", ["hard", "independent", "fragment_first", "soft"])
def test_full_chain_provider_and_loss(config, mode):
    config["semantic_coupling"] = mode
    chain = {
        "stages": [
            {
                "name": "reconstruction",
                "provider": "uresnet_multi",
                "uses": "uresnet_multi",
                "loss": "multi_loss",
                "adapt_labels": False,
            }
        ]
    }
    modules = {"uresnet_multi": config, "multi_loss": loss_config()}
    model = FullChain(chain, **modules).eval()
    data, seg, clusters, ppn = inputs()
    force_routing(model.uresnet_multi, seg.tensor)
    result = model(data)
    assert "fragment_clusts" in result and "ppn_points" in result
    objective = FullChainLoss(chain, **modules)
    metrics = objective(seg_label=seg, clust_label=clusters, ppn_label=ppn, **result)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()


def test_chain_loss_aligns_raw_truth_and_ignores_adapted_truth():
    data, seg, clusters, _ = inputs()
    seg = TensorBatch(
        torch.column_stack((data.tensor[:, :4], seg.tensor)),
        data.counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    selected = torch.tensor([1, 3, 65])
    index = IndexBatch(selected, data.counts, [2, 0, 1])
    captured = {}

    def loss(**kwargs):
        captured.update(kwargs)
        return {"loss": torch.tensor(0.0)}

    stage = MultiLossStage("multi", loss)
    stage.forward(
        {
            "orig_index": index,
            "segmentation": TensorBatch(torch.zeros(3, 5), [2, 0, 1]),
            "seg_label": seg,
            "clust_label": clusters,
            "clust_label_adapt": object(),
        }
    )
    torch.testing.assert_close(captured["seg_label"].tensor, seg.tensor[selected])
    torch.testing.assert_close(
        captured["seg_label"].values.torch_tensor(), seg.values.torch_tensor()[selected]
    )
    torch.testing.assert_close(
        captured["clust_label"].data.tensor, clusters.data.tensor[selected]
    )


def test_rejects_unsupported_coupling_and_embedded_ghost(config):
    with pytest.raises(ValueError, match="semantic_coupling"):
        UResNetMulti(**config, semantic_coupling="unknown")
    config["uresnet"]["ghost"] = True
    with pytest.raises(ValueError, match="deghosted"):
        UResNetMulti(**config)


@pytest.mark.parametrize("empty_input", [False, True])
def test_no_fragments_preserves_batch_metadata(config, empty_input):
    model = UResNetMulti(**config).eval()
    data, _, _, _ = inputs()
    if empty_input:
        data = TensorBatch(
            torch.empty((0, 5)), [0, 0, 0], has_batch_col=True, coord_cols=(1, 2, 3)
        )
    force_routing(model, torch.full((data.shape[0],), 4, dtype=torch.long))
    result = model(data)
    assert result["fragment_clusts"].counts.tolist() == [0, 0, 0]
    assert result["fragment_shapes"].shape[0] == 0
    assert result["seg_pred"].counts.tolist() == data.counts.tolist()


@pytest.mark.parametrize("mode", ["hard", "independent", "fragment_first", "soft"])
def test_modes_preserve_one_backbone_and_final_semantic_coherence(config, mode):
    """Every architecture publishes a uniform semantic class per fragment."""
    model = UResNetMulti(**config, semantic_coupling=mode).eval()
    data, seg, _, _ = inputs()
    classes = seg.tensor.clone()
    classes[::7] = 4
    force_routing(model, classes)
    if mode == "fragment_first":
        with torch.no_grad():
            model.fragment_head.classifier.bias[1] += 100

    result = model(data)
    assert sum(isinstance(m, UResNet) for m in model.modules()) == 1
    assert len(result["fragment_clusts"].index_list) > 0
    for cluster, shape in zip(
        result["fragment_clusts"].index_list, result["fragment_shapes"].tensor
    ):
        assert torch.all(result["seg_pred"].tensor[cluster] == int(shape))
        assert len(torch.unique(data.tensor[cluster, 0])) == 1
    torch.testing.assert_close(result["seg_pred_raw"].tensor, classes)

    if mode != "hard":
        assert result["graph_spice_filter_index"].index.tolist() == list(
            range(data.shape[0])
        )
        assert result["graph_spice_node_clusts"].counts.tolist() == [1, 1, 1]
        assert result["fragment_logits"].counts[1] == 0
    if mode == "fragment_first":
        assert torch.any(result["seg_pred"].tensor != classes)


@pytest.mark.parametrize("mode", ["independent", "fragment_first"])
def test_agnostic_edges_are_independent_of_voxel_semantic_predictions(config, mode):
    """Changing only voxel logits must not change an unconditioned graph."""
    model = UResNetMulti(**config, semantic_coupling=mode).eval()
    data, seg, _, _ = inputs()
    first_hook = force_routing(model, seg.tensor)
    first = model(data)
    first_hook.remove()
    force_routing(model, torch.full_like(seg.tensor, 4))
    second = model(data)

    for key in ("edge_index", "edge_attr"):
        torch.testing.assert_close(
            first[f"graph_spice_{key}"].tensor, second[f"graph_spice_{key}"].tensor
        )
    gradient = torch.autograd.grad(
        second["graph_spice_edge_attr"].tensor.sum(),
        second["segmentation"].tensor,
        allow_unused=True,
    )[0]
    assert gradient is None


@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("projected", [False, True])
def test_soft_edges_cross_class_boundaries_and_control_semantic_gradients(
    config, detach, projected
):
    """Soft routing keeps cross-class neighbors and optionally trains logits."""
    if projected:
        config["graph_spice"].update(
            {
                "use_raw_features": False,
                "embedder": {"feature_embedding_dim": 2},
                "kernel": {"provider": "default", "num_features": 2},
            }
        )
    model = UResNetMulti(
        **config, semantic_coupling="soft", semantic_conditioning={"detach": detach}
    ).eval()
    data, seg, _, _ = inputs()
    force_routing(model, seg.tensor)
    result = model(data)
    edges = result["graph_spice_edge_index"].tensor
    nodes = result["graph_spice_node_clusts"].index_list[0]
    edge_group = result["graph_spice_edge_clusts"].index_list[0]
    endpoints = nodes[edges[edge_group]]
    assert torch.any(seg.tensor[endpoints[:, 0]] != seg.tensor[endpoints[:, 1]])

    gradient = torch.autograd.grad(
        result["graph_spice_edge_attr"].tensor.sum(),
        result["segmentation"].tensor,
        allow_unused=True,
        retain_graph=True,
    )[0]
    if detach:
        assert gradient is None
    else:
        assert gradient is not None and gradient.abs().sum() > 0
    result["graph_spice_edge_attr"].tensor.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.uresnet_ppn.uresnet.backbone.parameters()
    )


@pytest.mark.parametrize("mode", ["independent", "fragment_first", "soft"])
def test_extended_modes_train_all_configured_heads(config, mode):
    """Actual joint losses must backpropagate through all new trainable heads."""
    model = UResNetMulti(**config, semantic_coupling=mode).eval()
    data, seg, clusters, ppn = inputs()
    result = model(data)
    loss = UResNetMultiLoss(
        **config,
        **loss_config(),
        semantic_coupling=mode,
        loss_balancing={"provider": "uncertainty"},
    )
    metrics = loss(seg, clusters, ppn, **result)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert all(p.grad is not None for p in model.graph_spice.kernel_fn.parameters())
    if mode == "fragment_first":
        assert metrics["fragment_semantics_count"] > 0
        assert all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.fragment_head.parameters()
        )


def test_learned_fragment_loss_uses_mixed_truth_and_excluded_candidates(config):
    """Rejected output classes must still receive fragment supervision."""
    model = UResNetMulti(**config, semantic_coupling="fragment_first").eval()
    with torch.no_grad():
        model.fragment_head.classifier.bias[4] += 100
    data, seg, clusters, ppn = inputs()
    result = model(data)
    assert len(result["fragment_clusts"].index_list) == 0
    assert len(result["fragment_candidates"].index_list) > 0
    loss = UResNetMultiLoss(
        **config, **loss_config(), semantic_coupling="fragment_first"
    )
    metrics = loss(seg, clusters, ppn, **result)
    assert metrics["fragment_semantics_count"] > 0
    metrics["fragment_semantics_loss"].backward()
    assert model.fragment_head.classifier.bias.grad.abs().sum() > 0

    # A mixed fragment targets its full class distribution, not a chosen label.
    from spine.model.uresnet.multi.semantics import FragmentSemanticLoss

    candidates = IndexBatch([torch.tensor([0, 1, 2])], [3], [1], [3])
    labels = TensorBatch(torch.tensor([0, 1, 1]), [3])
    logits = TensorBatch(torch.tensor([[0.5, -0.5]], requires_grad=True), [1])
    value = FragmentSemanticLoss()(labels, candidates, logits)
    expected = -(
        torch.log_softmax(logits.tensor, 1) * torch.tensor([[1 / 3, 2 / 3]])
    ).sum()
    torch.testing.assert_close(value["loss"], expected)
    assert value["purity"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mode", ["independent", "fragment_first", "soft"])
def test_extended_modes_handle_empty_batches(config, mode):
    """Empty candidate sets retain event counts and differentiable objectives."""
    data, seg, clusters, ppn = inputs()
    data = data.select(torch.zeros(data.shape[0], dtype=torch.bool))
    seg = seg.select(torch.zeros(seg.shape[0], dtype=torch.bool))
    clusters = clusters.select(torch.empty(0, dtype=torch.long), [0, 0, 0])
    ppn = ppn.select(torch.zeros(ppn.shape[0], dtype=torch.bool))
    model = UResNetMulti(**config, semantic_coupling=mode).eval()
    result = model(data)
    assert result["fragment_logits"].shape == (0, 5)
    assert result["fragment_clusts"].counts.tolist() == [0, 0, 0]
    loss = UResNetMultiLoss(**config, **loss_config(), semantic_coupling=mode)
    metrics = loss(seg, clusters, ppn, **result)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert all(p.grad is not None for p in model.graph_spice.kernel_fn.parameters())


def test_edge_targets_require_semantic_and_valid_cluster_agreement(config):
    """Equal raw IDs cannot join different semantics or two invalid clusters."""
    data, seg, clusters, _ = inputs()
    model = UResNetMulti(**config, semantic_coupling="independent").eval()
    result = model(data)
    graph = {
        key.removeprefix("graph_spice_"): value
        for key, value in result.items()
        if key.startswith("graph_spice_")
    }
    raw = clusters.data.tensor.clone()
    raw[:, 5] = 0
    raw[0, 5] = -1
    clusters = ClusterLabelBatch(
        TensorBatch(raw, data.counts, has_batch_col=True, coord_cols=(1, 2, 3))
    )
    loss = UResNetMultiLoss(**config, **loss_config(), semantic_coupling="independent")
    loss.label_graph(seg, clusters, graph)
    nodes = graph["node_clusts"].index_list[0]
    group = graph["edge_clusts"].index_list[0]
    endpoints = nodes[graph["edge_index"].tensor[group]]
    expected = (
        (seg.tensor[endpoints[:, 0]] == seg.tensor[endpoints[:, 1]])
        & (endpoints[:, 0] != 0)
        & (endpoints[:, 1] != 0)
    )
    assert torch.equal(graph["edge_label"].tensor[group], expected.long())


@pytest.mark.parametrize(
    "mode,policy",
    [("independent", "routed"), ("soft", "routed"), ("fragment_first", "pooled")],
)
def test_incompatible_semantic_policies_fail_early(config, mode, policy):
    """Configuration mistakes must not silently select another architecture."""
    with pytest.raises(ValueError):
        UResNetMulti(**config, semantic_coupling=mode, fragment_semantics=policy)


def test_agnostic_clustering_metrics_use_the_actual_target_partition(config):
    """Agnostic graph bookkeeping must not masquerade as a semantic metric."""
    data, seg, clusters, ppn = inputs()
    model = UResNetMulti(**config, semantic_coupling="independent").eval()
    result = model(data)
    losses = loss_config()
    losses["graph_spice_loss"]["evaluate_clustering_metrics"] = True
    loss = UResNetMultiLoss(**config, **losses, semantic_coupling="independent")
    metrics = loss(seg, clusters, ppn, **result)
    assert "graph_spice_ari" in metrics
    assert not any(key.startswith("graph_spice_ari_") for key in metrics)


@pytest.mark.parametrize("mode", ["independent", "soft"])
def test_learned_semantic_refinement_can_follow_parallel_fragmentation(config, mode):
    """Final classification is independently configurable for parallel heads."""
    data, seg, clusters, ppn = inputs()
    model = UResNetMulti(
        **config, semantic_coupling=mode, fragment_semantics="learned"
    ).eval()
    result = model(data)
    loss = UResNetMultiLoss(
        **config, **loss_config(), semantic_coupling=mode, fragment_semantics="learned"
    )
    metrics = loss(seg, clusters, ppn, **result)
    assert metrics["fragment_semantics_count"] > 0
    metrics["loss"].backward()
    assert model.fragment_head.classifier.weight.grad is not None


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"ppn": None}, TypeError, "PPN configuration mapping"),
        (
            {"semantic_conditioning": {"detach": True}},
            ValueError,
            "only valid for soft",
        ),
        (
            {"semantic_coupling": "soft", "semantic_conditioning": {"detach": 1}},
            TypeError,
            "detach must be a boolean",
        ),
        (
            {"semantic_coupling": "soft", "semantic_conditioning": {"typo": 1}},
            ValueError,
            "Unknown semantic_conditioning",
        ),
        ({"fragment_semantics": "unknown"}, ValueError, "Unknown fragment_semantics"),
    ],
)
def test_invalid_joint_options_fail_explicitly(config, options, error, message):
    """Reject malformed policies instead of silently changing the architecture."""
    config.update(options)
    with pytest.raises(error, match=message):
        UResNetMulti(**config)


@pytest.mark.parametrize("policy", ["pooled", "learned"])
def test_fragment_classification_requires_components(config, policy):
    """A classifier cannot train or infer without fragment membership."""
    config["graph_spice"]["make_clusters"] = False
    with pytest.raises(ValueError, match="require make_clusters=True"):
        UResNetMulti(**config, fragment_semantics=policy)


def test_fragment_classes_must_fit_voxel_classifier(config):
    """Reject output classes for which the shared semantic head has no logit."""
    config["uresnet"]["num_classes"] = 2
    with pytest.raises(ValueError, match="covered by the segmentation head"):
        UResNetMulti(**config)


@pytest.mark.parametrize("target", ["head", "loss"])
def test_empty_fragment_shape_configuration_is_rejected(config, target):
    """An agnostic graph still needs a nonempty output-class contract."""
    config["graph_spice"]["shapes"] = []
    with pytest.raises(ValueError, match="At least one fragment shape"):
        if target == "head":
            GraphSPICEHead(config["graph_spice"], 4, 4)
        else:
            UResNetMultiLoss(**config, **loss_config(), semantic_coupling="independent")


@pytest.mark.parametrize("option", ["uresnet", "coord_conv", "predict_semantics"])
def test_shared_graph_rejects_duplicate_backbone_or_semantics(config, option):
    """The graph head must consume the shared backbone and semantic outputs."""
    config["graph_spice"]["embedder"][option] = True
    message = "backbone inputs" if option != "predict_semantics" else "semantic head"
    with pytest.raises(ValueError, match=message):
        GraphSPICEHead(config["graph_spice"], 4, 4)


def test_semantic_conditioning_requires_agnostic_graph(config):
    """Soft evidence must not be combined with prior hard class filtering."""
    with pytest.raises(ValueError, match="requires an agnostic graph"):
        GraphSPICEHead(config["graph_spice"], 4, 4, semantic_channels=5)


@pytest.mark.parametrize("channels,width", [(0, 3), (0, 5), (5, None), (5, 3)])
def test_graph_head_validates_semantic_evidence(config, channels, width):
    """Reject missing or incompatible evidence before graph construction."""
    data, seg, _, _ = inputs()
    features = TensorBatch(torch.zeros(data.shape[0], 4), data.counts)
    probabilities = (
        None
        if width is None
        else TensorBatch(torch.zeros(data.shape[0], width), data.counts)
    )
    head = GraphSPICEHead(
        config["graph_spice"], 4, 4, class_agnostic=True, semantic_channels=channels
    )
    message = "requires class probabilities" if width is None else "probability width"
    with pytest.raises(ValueError, match=message):
        head(data, seg, features, probabilities)


@pytest.mark.parametrize("counts,width", [([0, 64, 64], 4), ([64, 0, 64], 3)])
def test_embedding_head_rejects_misaligned_features(counts, width):
    """Equal total row counts do not imply compatible event or channel axes."""
    data, _, _, _ = inputs()
    features = TensorBatch(torch.zeros(data.shape[0], width), counts)
    message = "batch counts" if width == 4 else "feature width"
    with pytest.raises(ValueError, match=message):
        GraphSPICEEmbeddingHead(4, 4)(data, features)


@pytest.mark.parametrize("option", ["num_classes", "dimension", "hidden_channels"])
@pytest.mark.parametrize("value", [0, -1])
def test_semantic_kernel_requires_positive_dimensions(option, value):
    """All conditioning dimensions must define a nonempty feature space."""
    from spine.model.graph_spice.kernel import SemanticKernel

    dimensions = {"num_classes": 5, "dimension": 3, "hidden_channels": 4}
    dimensions[option] = value
    with pytest.raises(ValueError, match="dimensions must be positive"):
        SemanticKernel(torch.nn.Identity(), **dimensions)


def test_fragment_head_rejects_routed_policy():
    """Routed classes are supplied by the graph, not a fragment classifier."""
    from spine.model.uresnet.multi.semantics import FragmentSemanticHead

    with pytest.raises(ValueError, match="supports pooled or learned"):
        FragmentSemanticHead(4, 5, "routed")


@pytest.mark.parametrize("adapt", [True, None, {"break_classes": []}])
def test_chain_provider_adapts_final_semantics_and_publishes_vertices(config, adapt):
    """Downstream truth adaptation sees final classes and preserves raw truth."""
    from spine.model.full_chain.providers.multi import build_multi_stage
    from spine.model.full_chain.state import ChainState

    config["vertex"] = {}
    config["semantic_coupling"] = "fragment_first"
    owner = torch.nn.Module()
    stage = build_multi_stage(
        "reconstruction", {"uresnet_multi": config, "adapt_labels": adapt}, owner
    )
    owner.eval()
    data, seg, clusters, _ = inputs()
    seg = TensorBatch(
        torch.column_stack((data.tensor[:, :4], seg.tensor)),
        data.counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    # Model predictions live on a selected domain and differ from raw routing.
    selected = torch.tensor([1, 3, 65])
    index = IndexBatch(selected, data.counts, [2, 0, 1])
    current = TensorBatch(
        data.tensor[selected], [2, 0, 1], has_batch_col=True, coord_cols=(1, 2, 3)
    )
    force_routing(stage.model, torch.zeros(3, dtype=torch.long))
    with torch.no_grad():
        stage.model.fragment_head.classifier.bias[1] += 100
    original = clusters.data.tensor.clone()
    state = ChainState(
        data=current, orig_index=index, seg_label=seg, clust_label=clusters
    )
    result = stage(state)
    assert owner.uresnet_multi is stage.model
    assert torch.all(result.outputs["seg_pred"].tensor == 1)
    assert torch.all(result.outputs["seg_pred_raw"].tensor == 0)
    assert result.products["vertex_proposals"] is result.outputs["vertex_points"]
    assert result.products["clust_label"] is result.outputs["clust_label_adapt"]
    expected = stage.label_adapter(
        clusters,
        seg,
        result.outputs["seg_pred"],
        orig_index=state.get("point_data").orig_index,
    )
    torch.testing.assert_close(result.products["clust_label"].tensor, expected.tensor)
    torch.testing.assert_close(clusters.data.tensor, original)


def test_chain_provider_requires_fragments(config):
    """A stage promising canonical fragments must actually construct them."""
    from spine.model.full_chain.providers.multi import build_multi_stage

    config["graph_spice"]["make_clusters"] = False
    with pytest.raises(ValueError, match="requires make_clusters=True"):
        build_multi_stage("multi", {"uresnet_multi": config}, torch.nn.Module())


def test_chain_provider_rejects_invalid_adaptation_policy(config):
    """Reject unsupported adaptation settings instead of enabling defaults."""
    from spine.model.full_chain.providers.multi import build_multi_stage

    with pytest.raises(TypeError, match="boolean or mapping"):
        build_multi_stage(
            "multi", {"uresnet_multi": config, "adapt_labels": "yes"}, torch.nn.Module()
        )


def test_inference_chain_does_not_register_an_objective():
    """An unsupervised stage does not require model/loss configuration blocks."""
    from spine.model.full_chain.providers.multi import build_multi_loss

    owner = torch.nn.Module()
    assert build_multi_loss("multi", {}, owner) is None
    assert not list(owner.children())


def test_standalone_graph_keeps_label_free_training_contract(config):
    """The backbone refactor preserves standalone predictions and loss targets."""
    from spine.model.graph_spice import GraphSPICE, GraphSPICELoss

    graph_config = config["graph_spice"]
    graph_config["embedder"]["coord_conv"] = False
    graph_config["embedder"]["uresnet"] = {
        key: value for key, value in config["uresnet"].items() if key != "num_classes"
    }
    model = GraphSPICE(graph_config).eval()
    data, seg, clusters, _ = inputs()
    predictions = model(data, seg)
    legacy = model(data, seg, clusters)
    assert "edge_label" not in predictions
    torch.testing.assert_close(
        predictions["edge_attr"].tensor, legacy["edge_attr"].tensor
    )
    objective = GraphSPICELoss(graph_config, loss_config()["graph_spice_loss"])
    metrics = objective(seg, clusters, **predictions)
    expected = objective(seg, clusters, **legacy)
    torch.testing.assert_close(metrics["loss"], expected["loss"])
    metrics["loss"].backward()
    assert any(p.grad is not None for p in model.embedder.backbone.parameters())
