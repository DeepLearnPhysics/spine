"""Tests for modular image encoders and prediction heads."""

import pytest

torch = pytest.importorskip("torch")

from spine.data import IndexBatch, TensorBatch
from spine.model.image.encoder import (
    ImageCNNEncoder,
    ImagePointNetEncoder,
    image_encoder_factory,
)
from spine.model.image.model import ImageModel


class DummyEncoder(torch.nn.Module):
    """Small deterministic encoder implementing the image encoder contract."""

    feature_size = 2

    def forward(self, data):
        """Return the mean and sum of the final input column per object."""
        table = data.torch_tensor()
        values = table[:, -1]
        features = []
        for batch_id in range(data.batch_size):
            selected = values[data.batch_ids == batch_id]
            features.append(torch.stack((selected.mean(), selected.sum())))
        return torch.stack(features)


def test_image_model_supports_multiple_named_heads(monkeypatch, image_data):
    """One shared object encoder should feed classification and regression."""
    monkeypatch.setattr(
        "spine.model.image.model.image_encoder_factory",
        lambda _cfg: DummyEncoder(),
    )
    model = ImageModel(
        {
            "objects": {"source": "image"},
            "encoder": {"name": "dummy"},
            "heads": {"pid": 5, "energy": 1},
            "return_features": True,
        }
    )

    result = model(image_data)

    assert result["objects"].counts.tolist() == [1, 1]
    assert result["features"].shape == (2, 2)
    assert result["pid_pred"].shape == (2, 5)
    assert result["energy_pred"].shape == (2, 1)
    assert result["pid_pred"].counts.tolist() == [1, 1]


def test_image_model_rejects_reserved_head_name(monkeypatch):
    """Dynamic head names cannot overwrite structural model products."""
    monkeypatch.setattr(
        "spine.model.image.model.image_encoder_factory",
        lambda _cfg: DummyEncoder(),
    )
    with pytest.raises(ValueError, match="reserved"):
        ImageModel(
            {
                "encoder": {"name": "dummy"},
                "heads": {"objects": 2},
            }
        )


def test_pointnet_adapter_consumes_objectized_tensor_batches():
    """PointNet should implement the same interface as the sparse CNN."""
    data = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    batch = TensorBatch(data, counts=torch.tensor([3, 3]))
    encoder = ImagePointNetEncoder(
        depth=1,
        sampling_ratio=1.0,
        neighbor_radius=10.0,
        mlp_specs_0=[4, 8],
        mlp_specs_glob=[11, 8],
        mlp_specs_final=[8, 4],
        dropout=0.0,
    )

    output = encoder(batch)

    assert output.shape == (2, 4)
    assert encoder.feature_size == 4


def test_image_encoder_factory_validates_and_constructs_adapters():
    """The image factory requires a known named adapter without mutation."""
    pointnet = {
        "name": "pointnet",
        "depth": 1,
        "sampling_ratio": 1.0,
        "neighbor_radius": 10.0,
        "mlp_specs_0": [4, 8],
        "mlp_specs_glob": [11, 8],
        "mlp_specs_final": [8, 4],
        "dropout": 0.0,
    }
    assert isinstance(image_encoder_factory(pointnet), ImagePointNetEncoder)
    assert pointnet["name"] == "pointnet"

    cnn = {
        "name": "cnn",
        "reps": 1,
        "depth": 2,
        "filters": 4,
        "num_input": 1,
        "data_dim": 3,
        "activation": "relu",
        "norm_layer": "none",
        "spatial_size": 4,
        "feature_size": 8,
    }
    assert isinstance(image_encoder_factory(cnn), ImageCNNEncoder)

    with pytest.raises(ValueError, match="requires `name`"):
        image_encoder_factory({})
    with pytest.raises(ValueError, match="Unknown image encoder"):
        image_encoder_factory({"name": "transformer"})


def test_pointnet_image_adapter_validates_input_layout():
    """The current PointNet adapter accepts only featured 3D point clouds."""
    with pytest.raises(ValueError, match="requires 3D"):
        ImagePointNetEncoder(data_dim=2)
    with pytest.raises(ValueError, match="num_input"):
        ImagePointNetEncoder(num_input=0)


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"heads": {"pid": 2}}, ValueError, "requires `encoder`"),
        ({"encoder": {}}, ValueError, "requires `heads`"),
        (
            {"encoder": {}, "heads": {"pid": 2}, "unknown": True},
            ValueError,
            "Unknown image model options",
        ),
        (
            {"objects": [], "encoder": {}, "heads": {"pid": 2}},
            TypeError,
            "objects.*mapping",
        ),
        (
            {"encoder": [], "heads": {"pid": 2}},
            TypeError,
            "encoder.*mapping",
        ),
        ({"encoder": {}, "heads": {}}, ValueError, "nonempty mapping"),
        ({"encoder": {}, "heads": {"": 2}}, ValueError, "nonempty strings"),
        ({"encoder": {}, "heads": {"pid": 0}}, ValueError, "positive"),
        (
            {"encoder": {}, "heads": {"pid": {"name": "linear"}}},
            ValueError,
            "requires `out_channels`",
        ),
        (
            {"encoder": {}, "heads": {"pid": {"out_channels": 0}}},
            ValueError,
            "positive",
        ),
        (
            {"encoder": {}, "heads": {"pid": []}},
            TypeError,
            "integer or mapping",
        ),
    ],
)
def test_image_model_validates_modular_configuration(
    monkeypatch,
    config,
    error,
    message,
):
    """Malformed object, encoder, and head blocks fail during construction."""
    monkeypatch.setattr(
        "spine.model.image.model.image_encoder_factory",
        lambda _cfg: DummyEncoder(),
    )
    with pytest.raises(error, match=message):
        ImageModel(config)


def test_image_model_handles_empty_object_batches(monkeypatch, image_data):
    """Empty reconstructed object sets bypass the encoder with typed outputs."""
    monkeypatch.setattr(
        "spine.model.image.model.image_encoder_factory",
        lambda _cfg: DummyEncoder(),
    )
    model = ImageModel(
        {
            "objects": {"source": "explicit"},
            "encoder": {},
            "heads": {"pid": 2},
            "return_features": True,
        }
    )
    objects = IndexBatch(
        [],
        spans=image_data.counts,
        counts=[0, 0],
        single_counts=[],
    )

    result = model(image_data, objects=objects)

    assert result["features"].shape == (0, 2)
    assert result["pid_pred"].shape == (0, 2)
    assert result["pid_pred"].counts.tolist() == [0, 0]


def test_image_model_rejects_encoder_contract_violation(monkeypatch, image_data):
    """Encoders must return exactly one configured-width vector per object."""

    class BadEncoder(DummyEncoder):
        def forward(self, data):
            return torch.zeros((data.batch_size, 3))

    monkeypatch.setattr(
        "spine.model.image.model.image_encoder_factory",
        lambda _cfg: BadEncoder(),
    )
    model = ImageModel(
        {"objects": {"source": "image"}, "encoder": {}, "heads": {"pid": 2}}
    )

    with pytest.raises(RuntimeError, match="Image encoder returned shape"):
        model(image_data)
