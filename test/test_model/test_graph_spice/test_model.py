"""Behavioral tests for the top-level GraphSPICE model and loss."""

from copy import deepcopy
from typing import Any, cast

import pytest
import torch

from spine.constants import SHOWR_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch, TensorSchema
from spine.model.graph_spice import EdgeLoss, GraphSPICE, GraphSPICELoss
from spine.model.graph_spice.constructor import ClusterGraphConstructor


def test_graph_spice_rejects_misaligned_segmentation_labels():
    """Filtering must validate labels before applying their indexes to data."""
    model = object.__new__(GraphSPICE)
    model.shapes = [0]
    data = TensorBatch(torch.zeros((2, 5)), counts=[2], has_batch_col=True)
    seg_label = TensorBatch(torch.zeros(3), counts=[3])

    with pytest.raises(ValueError, match="matching row counts"):
        model.filter_class(data, seg_label)


def test_graph_spice_loss_builds_explicit_loss_configuration():
    """The required loss block must resolve its named implementation."""
    loss = GraphSPICELoss(
        {"constructor": {}},
        {"name": "edge", "metric": None},
    )

    assert isinstance(loss.loss_fn, EdgeLoss)
    assert loss.constructor is None


def test_graph_spice_loss_requires_loss_configuration():
    """A Graph-SPICE model block alone cannot define its training loss."""
    # Deliberately bypass static argument checking to exercise the runtime
    # failure produced by the model manager when the configuration omits this
    # required block.
    loss_cls: Any = GraphSPICELoss
    with pytest.raises(TypeError, match="graph_spice_loss"):
        loss_cls({"constructor": {}})


def test_loss_edge_labels_match_network_labels():
    """Loss-side labeling must match optional network-side targets."""
    constructor_cfg = {
        "graph": {"name": "radius", "r": 1.9},
        "shapes": ["shower"],
        "edge_threshold": 0.1,
        "label_edges": True,
    }
    constructor = ClusterGraphConstructor(**deepcopy(constructor_cfg))
    constructor.graph_fn = lambda coords: torch.tensor(
        [[0], [1]],
        dtype=torch.long,
        device=coords.device,
    )
    constructor.kernel_fn = lambda left, right: torch.zeros(
        len(left),
        dtype=left.dtype,
        device=left.device,
    )

    coords = TensorBatch(torch.zeros((2, 3)), counts=[2])
    features = TensorBatch(torch.zeros((2, 2)), counts=[2])
    seg_tensor = torch.full((2, 1), SHOWR_SHP)
    seg_label = TensorBatch(seg_tensor, counts=[2])
    clust_tensor = torch.zeros((2, 6))
    clust_tensor[:, 5] = 7
    clust_label = ClusterLabelBatch(
        TensorBatch(
            clust_tensor,
            counts=[2],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        )
    )
    cluster_ids = clust_label.voxel_field("cluster")

    labeled_graph = constructor(coords, features, seg_label, cluster_ids)
    label_free_graph = constructor(coords, features, seg_label)
    loss = GraphSPICELoss(
        {"constructor": constructor_cfg},
        {"name": "edge", "metric": None},
    )
    edge_index = cast(TensorBatch, label_free_graph["edge_index"])
    node_clusts = cast(IndexBatch, label_free_graph["node_clusts"])
    edge_clusts = cast(IndexBatch, label_free_graph["edge_clusts"])
    derived_labels = loss.get_edge_labels(
        clust_label,
        edge_index,
        node_clusts,
        edge_clusts,
    )

    assert "edge_label" not in label_free_graph
    network_labels = cast(TensorBatch, labeled_graph["edge_label"])
    assert torch.equal(
        derived_labels.torch_tensor(),
        network_labels.torch_tensor(),
    )


def _graph_spice_inputs():
    """Build two aligned voxel rows and structured cluster truth."""
    rows = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 2.0],
        ]
    )
    data = TensorBatch(
        rows,
        counts=[2],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    seg_label = TensorBatch(torch.full((2,), SHOWR_SHP), counts=[2])
    compact = torch.column_stack((rows, torch.tensor([4.0, 4.0])))
    clust_label = ClusterLabelBatch(
        TensorBatch(
            compact,
            counts=[2],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        )
    )

    return data, seg_label, clust_label


def test_graph_spice_filters_structured_labels_and_validates_rows():
    """Class filtering keeps structured labels aligned and checks their rows."""
    model = object.__new__(GraphSPICE)
    model.shapes = [SHOWR_SHP]
    data, seg_label, clust_label = _graph_spice_inputs()

    filtered = model.filter_class(data, seg_label, clust_label)
    assert filtered[2].data.shape[0] == 2
    assert filtered[3].index.tolist() == [0, 1]

    short_label = ClusterLabelBatch(
        TensorBatch(
            clust_label.data.torch_tensor()[:1],
            counts=[1],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        )
    )
    with pytest.raises(ValueError, match="cluster label tensors"):
        model.filter_class(data, seg_label, short_label)


def test_graph_spice_forward_selects_features_and_builds_clusters():
    """Graph construction consumes embeddings, truth IDs and optional fitting."""
    data, seg_label, clust_label = _graph_spice_inputs()

    class Embedder(torch.nn.Module):
        def forward(self, selected):
            features = TensorBatch(torch.ones((2, 2)), selected.counts)
            embeddings = TensorBatch(torch.full((2, 3), 2.0), selected.counts)
            return {
                "coordinates": selected,
                "features": features,
                "hypergraph_features": embeddings,
            }

    class Constructor:
        target_col = "cluster"

        def __init__(self):
            self.received = None

        def __call__(self, coords, features, labels, cluster_ids):
            self.received = (coords, features, labels, cluster_ids)
            return {"edge_pred": TensorBatch(torch.zeros(1), counts=[1])}

        def fit_predict(self, graph):
            return IndexBatch([torch.tensor([0, 1])], [2], [1], [2]), [SHOWR_SHP]

    model = object.__new__(GraphSPICE)
    torch.nn.Module.__init__(model)
    model.shapes = [SHOWR_SHP]
    model.embedder = Embedder()
    model.constructor = Constructor()
    model.use_raw_features = False
    model.make_clusters = True

    result = model(data, seg_label, clust_label)

    assert model.constructor.received[1].torch_tensor()[0, 0] == 2.0
    assert torch.equal(
        model.constructor.received[3].torch_tensor(),
        torch.tensor([4.0, 4.0]),
    )
    assert "clusts" in result
    assert result["clust_shapes"] == [SHOWR_SHP]

    model.use_raw_features = True
    model.make_clusters = False
    model(data, seg_label)
    assert model.constructor.received[1].torch_tensor()[0, 0] == 1.0
    assert model.constructor.received[3] is None


def test_graph_spice_forward_requires_coordinate_schema():
    """The embedder must identify coordinate columns for graph construction."""
    data, seg_label, _ = _graph_spice_inputs()

    class Embedder(torch.nn.Module):
        def forward(self, selected):
            coordinates = TensorBatch(
                selected.torch_tensor(),
                selected.counts,
                schema=TensorSchema(),
            )
            features = TensorBatch(torch.ones((2, 2)), selected.counts)
            return {
                "coordinates": coordinates,
                "features": features,
                "hypergraph_features": features,
            }

    model = object.__new__(GraphSPICE)
    torch.nn.Module.__init__(model)
    model.shapes = [SHOWR_SHP]
    model.embedder = Embedder()
    model.constructor = lambda *args: {}
    model.use_raw_features = True
    model.make_clusters = False

    with pytest.raises(RuntimeError, match="coordinate columns"):
        model(data, seg_label)


def test_graph_spice_edge_label_group_validation():
    """Loss-side edge labeling rejects malformed semantic graph partitions."""
    _, _, clust_label = _graph_spice_inputs()
    loss = GraphSPICELoss(
        {"constructor": {}},
        {"name": "edge", "metric": None},
    )
    edges = TensorBatch(torch.tensor([[0, 1]]), counts=[1])

    direct = IndexBatch(torch.tensor([0, 1]), spans=[2], counts=[2])
    grouped = IndexBatch(
        [torch.tensor([0, 1])],
        spans=[2],
        counts=[1],
        single_counts=[2],
    )
    edge_groups = IndexBatch(
        [torch.tensor([0])],
        spans=[1],
        counts=[1],
        single_counts=[1],
    )
    with pytest.raises(TypeError, match="index lists"):
        loss.get_edge_labels(clust_label, edges, direct, edge_groups)

    extra_groups = IndexBatch(
        [torch.tensor([0]), torch.tensor([1])],
        spans=[2],
        counts=[2],
        single_counts=[1, 1],
    )
    with pytest.raises(ValueError, match="same number"):
        loss.get_edge_labels(clust_label, edges, grouped, extra_groups)

    numpy_groups = IndexBatch(
        [torch.tensor([0, 1]).numpy()],
        spans=[2],
        counts=[1],
        single_counts=[2],
    )
    with pytest.raises(TypeError, match="PyTorch tensors"):
        loss.get_edge_labels(clust_label, edges, numpy_groups, edge_groups)

    bad_edges = TensorBatch(torch.tensor([[0, 2]]), counts=[1])
    with pytest.raises(IndexError, match="outside"):
        loss.get_edge_labels(clust_label, bad_edges, grouped, edge_groups)

    uncovered = IndexBatch(
        [torch.empty(0, dtype=torch.long)],
        spans=[1],
        counts=[1],
        single_counts=[0],
    )
    with pytest.raises(ValueError, match="cover every edge"):
        loss.get_edge_labels(clust_label, edges, grouped, uncovered)


def test_graph_spice_loss_evaluates_clustering_metrics(monkeypatch):
    """Metric mode builds missing node predictions and merges graph metrics."""

    class Constructor:
        def __init__(self, **kwargs):
            self.fitted = False

        def fit_predict(self, output):
            self.fitted = True

        def evaluate(self, output, mean):
            assert mean
            return {"ari": 0.75}

    monkeypatch.setattr(
        "spine.model.graph_spice.model.ClusterGraphConstructor",
        Constructor,
    )
    loss = GraphSPICELoss(
        {"constructor": {}},
        {"name": "edge", "metric": None, "evaluate_clustering_metrics": True},
    )

    class Loss(torch.nn.Module):
        def forward(self, **kwargs):
            return {"loss": torch.tensor(0.0)}

    loss.loss_fn = Loss()
    data, seg_label, clust_label = _graph_spice_inputs()
    filter_index = IndexBatch(torch.tensor([0, 1]), spans=[2], counts=[2])

    result = loss(
        seg_label,
        clust_label,
        filter_index,
        edge_label=TensorBatch(torch.ones(1), counts=[1]),
    )

    assert loss.constructor.fitted
    assert result["ari"] == 0.75
