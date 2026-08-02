"""Modular whole-image and object-image prediction models."""

from .encoder import (
    ImageCNNEncoder,
    ImageEncoder,
    ImagePointNetEncoder,
    image_encoder_factory,
)
from .loss import (
    ImageClassificationLoss,
    ImageLoss,
    ImageRegressionLoss,
    ImageTaskLoss,
)
from .model import MODEL_SPEC, ImageModel
from .object import ImageObjectBuilder

__all__ = [
    "ImageModel",
    "ImageLoss",
    "ImageObjectBuilder",
    "ImageEncoder",
    "ImageCNNEncoder",
    "ImagePointNetEncoder",
    "ImageClassificationLoss",
    "ImageRegressionLoss",
    "ImageTaskLoss",
    "image_encoder_factory",
    "MODEL_SPEC",
]
