"""Shared helpers for torch-backed datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from spine.utils.conditional import TORCH_AVAILABLE

from ..augment import AugmentManager

if TORCH_AVAILABLE:
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Import-safe stand-in used when PyTorch is unavailable."""

        pass


DataDict = dict[str, Any]
Augmenter = Callable[[DataDict], DataDict]


class BaseDataset(Dataset):
    """Shared behavior for SPINE torch datasets."""

    _index_keys: ClassVar[tuple[str, str, str]] = (
        "index",
        "file_index",
        "file_entry_index",
    )
    augmenter: Augmenter | None

    def __init__(self) -> None:
        """Initialize shared dataset state."""
        self.augmenter = None

    def build_augmenter(
        self, augment: Mapping[str, Any] | None, geo: Mapping[str, Any] | None = None
    ) -> None:
        """Instantiate the configured augmenter, if any."""
        if augment is None:
            self.augmenter = None
            return

        kwargs = dict(augment)
        if geo is not None:
            self.augmenter = AugmentManager(geo=geo, **kwargs)
        else:
            self.augmenter = AugmentManager(**kwargs)

    def apply_augmenter(self, data: DataDict) -> DataDict:
        """Apply the configured augmenter, if present."""
        if self.augmenter is None:
            return data

        return self.augmenter(data)

    @classmethod
    def index_data_types(cls) -> dict[str, str]:
        """Return the standard scalar types for dataset index keys."""
        return {key: "scalar" for key in cls._index_keys}

    @classmethod
    def index_overlay_methods(cls) -> dict[str, str]:
        """Return the standard overlay methods for dataset index keys."""
        return {key: "cat" for key in cls._index_keys}
