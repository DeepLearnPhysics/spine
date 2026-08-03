"""End-to-end layer tests for the maintained SPICE implementation."""

import itertools

import pytest
import torch

from spine.data import ClusterLabelBatch, TensorBatch
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
