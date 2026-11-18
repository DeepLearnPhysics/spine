"""Shared UResNet semantic, point-proposal and fragmentation model."""

from .model import MODEL_SPEC, UResNetMulti, UResNetMultiLoss

__all__ = ["UResNetMulti", "UResNetMultiLoss", "MODEL_SPEC"]
