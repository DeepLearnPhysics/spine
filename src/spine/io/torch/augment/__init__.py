"""Data augmentation managers and modules."""

from .base import AugmentBase
from .crop import CropAugment
from .jitter import JitterAugment
from .manager import AugmentManager
from .mask import MaskAugment
from .rotate import RotateAugment
from .translate import TranslateAugment

Augmenter = AugmentManager

__all__ = [
    "AugmentManager",
    "Augmenter",
    "AugmentBase",
    "CropAugment",
    "JitterAugment",
    "MaskAugment",
    "RotateAugment",
    "TranslateAugment",
]
