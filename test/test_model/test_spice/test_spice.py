"""End-to-end layer tests for the maintained SPICE implementation."""

import itertools

import pytest
import torch

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.spice import SPICEEmbedder, SPICELoss


def spice_config():
    """Return a minimal SPICE model configuration."""
    return {
        "uresnet": {
            "reps": 1,
            "depth": 2,
            "filters": 4,
            "num_input": 4,
            "data_dim": 3,
            "activation": "relu",
            "norm_layer": "none",
            "spatial_size": 4,
        },
        "skip_classes": ["michel"],
        "coord_conv": True,
    }


def spice_batch():
    """Build one point cloud with two clusters and one excluded class."""
    data_rows = []
    semantic_rows = []
    label_rows = []
    for point in itertools.product(range(4), repeat=3):
        x, y, z = point
        data_rows.append((0, x, y, z, float(x + y + z + 1)))

        cluster_id = int(x >= 2)
        shape = 2 if z == 3 else 0
        particle_index = 2 * cluster_id + int(shape == 2)
        label = [0, x, y, z, float(x + y + z + 1), cluster_id, particle_index]
        label_rows.append(label)
        semantic_rows.append((0, x, y, z, shape))

    data_tensor = torch.tensor(data_rows, dtype=torch.float32)
    semantic_tensor = torch.tensor(semantic_rows, dtype=torch.float32)
    label_tensor = torch.tensor(label_rows, dtype=torch.float32)
    return (
        TensorBatch(
            data_tensor,
            counts=[len(data_tensor)],
            has_batch_col=True,
            coord_cols=(1, 2, 3),
        ),
        TensorBatch(
            semantic_tensor[:, -1],
            counts=[len(semantic_tensor)],
        ),
        ClusterLabelBatch(
            TensorBatch(label_tensor, counts=[len(label_tensor)], has_batch_col=True),
            {"shape": TensorBatch(torch.tensor([0, 2, 0, 2]), counts=[4])},
        ),
    )


def test_spice_embedder_filters_and_predicts_current_contract():
    """SPICE must return aligned TensorBatch outputs for retained voxels."""
    data, semantics, _ = spice_batch()
    embedder = SPICEEmbedder(**spice_config())

    result = embedder(data, semantics)

    retained = 48
    assert len(result["filter_index"].index) == retained
    assert result["embeddings"].shape == (retained, 3)
    assert result["margins"].shape == (retained, 1)
    assert result["seediness"].shape == (retained, 1)
    assert torch.all(result["margins"].torch_tensor() > 0.0)
    assert torch.all((result["seediness"].torch_tensor() >= 0.0))
    assert torch.all((result["seediness"].torch_tensor() <= 1.0))


def test_spice_loss_is_finite_and_differentiable():
    """The current model and objective must support a complete backward pass."""
    data, semantics, labels = spice_batch()
    embedder = SPICEEmbedder(**spice_config())
    loss_fn = SPICELoss(spice_config(), {"min_voxels": 2})

    output = embedder(data, semantics)
    result = loss_fn(clust_label=labels, **output)

    assert result["count"] == 48
    assert torch.isfinite(result["loss"])
    assert 0.0 <= result["accuracy"] <= 1.0
    result["loss"].backward()
    assert any(
        parameter.grad is not None
        for parameter in embedder.parameters()
        if parameter.requires_grad
    )


def test_spice_preserves_batch_entries_with_no_retained_voxels():
    """Semantic filtering must retain trailing empty batch metadata."""
    data, semantics, _ = spice_batch()
    second_data = data.torch_tensor().clone()
    second_data[:, 0] = 1
    second_semantics = torch.full_like(semantics.torch_tensor(), 2)
    data = TensorBatch(
        torch.cat((data.torch_tensor(), second_data)),
        counts=[64, 64],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    semantics = TensorBatch(
        torch.cat((semantics.torch_tensor(), second_semantics)),
        counts=[64, 64],
    )

    result = SPICEEmbedder(**spice_config())(data, semantics)

    assert result["embeddings"].counts.tolist() == [48, 0]
    assert result["filter_index"].counts.tolist() == [48, 0]


def test_spice_validates_coordinate_convolution_input_width():
    """UResNet feature width must agree with coordinate convolution."""
    config = spice_config()
    config["uresnet"]["num_input"] = 1

    with pytest.raises(ValueError, match="expected `num_input=4`"):
        SPICEEmbedder(**config)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"spatial_size": None}, "requires `uresnet.spatial_size`"),
        ({"spatial_size": 0}, "spatial_size.*positive"),
    ],
)
def test_spice_validates_spatial_extent(update, message):
    """Embedding-coordinate normalization requires a positive detector extent."""
    config = spice_config()
    config["uresnet"].update(update)
    with pytest.raises(ValueError, match=message):
        SPICEEmbedder(**config)


@pytest.mark.parametrize("kwargs", [{"margin_dim": 2}, {"seediness_dim": 2}])
def test_spice_validates_scalar_output_dimensions(kwargs):
    """The maintained objective accepts one margin and seed score per voxel."""
    with pytest.raises(ValueError, match="exactly one"):
        SPICEEmbedder(**spice_config(), **kwargs)


def test_spice_seed_freeze_and_coordinate_free_frontend():
    """The seed branch can be frozen and coordinate convolution disabled."""
    config = spice_config()
    config["uresnet"]["num_input"] = 1
    config["coord_conv"] = False
    embedder = SPICEEmbedder(**config, seed_freeze=True)

    assert not any(
        parameter.requires_grad for parameter in embedder.seediness_decoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in embedder.seediness_output.parameters()
    )
    data, semantics, _ = spice_batch()
    assert embedder(data, semantics)["embeddings"].shape == (48, 3)


def test_spice_filter_and_feature_contracts():
    """Semantic filtering requires aligned rows and at least one input feature."""
    data, semantics, _ = spice_batch()
    embedder = SPICEEmbedder(**spice_config())
    with pytest.raises(ValueError, match="same length"):
        embedder.filter_class(data, TensorBatch(semantics.tensor[:-1], [63]))

    coordinate_only = TensorBatch(
        data.tensor[:, :4],
        counts=data.counts,
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="at least one input feature"):
        embedder(coordinate_only, semantics)


@pytest.mark.parametrize(
    ("spice", "loss", "message"),
    [
        ({"margin_dim": 2}, {}, "requires one margin"),
        ({}, {"embedding_weight": -1.0}, "weights must be nonnegative"),
        ({}, {"inter_margin": -1.0}, "inter_margin"),
        ({}, {"min_voxels": 0}, "min_voxels"),
        ({}, {"eps": 0.0}, "eps"),
    ],
)
def test_spice_loss_validates_configuration(spice, loss, message):
    """Malformed SPICE model/loss numerical contracts fail at construction."""
    with pytest.raises(ValueError, match=message):
        SPICELoss(spice, loss)


def test_spice_loss_handles_unsupervised_and_misaligned_outputs():
    """Sparse classes yield differentiable zero and row mismatches fail."""
    _, _, labels = spice_batch()
    loss_fn = SPICELoss({}, {"min_voxels": 3})
    embeddings = TensorBatch(torch.zeros((2, 3), requires_grad=True), [2])
    margins = TensorBatch(torch.ones((2, 1)), [2])
    seediness = TensorBatch(torch.zeros((2, 1)), [2])
    indexes = IndexBatch(torch.tensor([0, 1]), spans=[64], counts=[2])

    result = loss_fn(labels, embeddings, margins, seediness, indexes)
    assert result["count"] == 0
    assert result["accuracy"] == 1.0
    result["loss"].backward()

    bad_seediness = TensorBatch(torch.zeros((1, 1)), [1])
    with pytest.raises(ValueError, match="equal length"):
        loss_fn(labels, embeddings, margins, bad_seediness, indexes)
