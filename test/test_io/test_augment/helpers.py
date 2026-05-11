"""Shared helpers for torch augmenter tests."""

from typing import Any

import numpy as np
import pytest

from spine.data import Meta
from spine.geo import GeoManager
from spine.io.parse.data import ParserTensor

try:
    from spine.io.augment import (
        CropAugment,
        FlipAugment,
        MaskAugment,
        RotateAugment,
    )
except ImportError as err:  # pragma: no cover - environment dependent
    pytest.skip(f"Torch augmentation stack unavailable: {err}", allow_module_level=True)


BOX2 = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)


def make_meta(lower=(0.0, 0.0, 0.0), upper=(2.0, 4.0, 2.0), size=(1.0, 1.0, 1.0)):
    """Build a simple 3D metadata object."""
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    count = np.rint((upper - lower) / size).astype(np.int64)
    return Meta(lower=lower, upper=upper, size=size, count=count)


def make_tensor(coords, meta):
    """Build a sparse parser tensor for augmentation tests."""
    coords = np.asarray(coords, dtype=np.int64)
    features = np.arange(len(coords), dtype=np.float32).reshape(-1, 1)
    return ParserTensor(coords=coords.copy(), features=features, meta=meta)


__all__ = [
    "Any",
    "BOX2",
    "CropAugment",
    "FlipAugment",
    "GeoManager",
    "MaskAugment",
    "RotateAugment",
    "ParserTensor",
    "make_meta",
    "make_tensor",
    "np",
    "pytest",
]
