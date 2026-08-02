"""Tests for modular image encoders and prediction heads."""

import pytest

torch = pytest.importorskip("torch")

from spine.data import TensorBatch
from spine.model.image.encoder import ImagePointNetEncoder
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
