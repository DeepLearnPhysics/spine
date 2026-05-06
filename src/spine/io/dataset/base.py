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
    """Shared behavior for SPINE torch datasets.

    This base class centralizes the small amount of logic that every SPINE
    dataset needs:

    - construction of an optional augmenter
    - consistent extraction of reader-produced metadata
    - default collate-type and overlay behavior for index/provenance fields

    Concrete dataset classes remain responsible for instantiating their
    backend reader and converting raw reader outputs into parser products.
    """

    _index_keys: ClassVar[tuple[str, str, str]] = (
        "index",
        "file_index",
        "file_entry_index",
    )
    _source_keys: ClassVar[tuple[str, ...]] = (
        "source_file_name",
        "source_file_size",
        "source_file_mtime_ns",
        "source_file_entry_index",
    )
    augmenter: Augmenter | None

    def __init__(self) -> None:
        """Initialize shared dataset state."""
        self.augmenter = None

    def build_augmenter(
        self, augment: Mapping[str, Any] | None, geo: Mapping[str, Any] | None = None
    ) -> None:
        """Instantiate the configured augmenter, if any.

        Parameters
        ----------
        augment : mapping, optional
            Augmentation configuration block passed to
            :class:`spine.io.augment.AugmentManager`.
        geo : mapping, optional
            Geometry configuration forwarded to the augmenter when geometric
            augmentations are enabled.
        """
        if augment is None:
            self.augmenter = None
            return

        kwargs = dict(augment)
        if geo is not None:
            self.augmenter = AugmentManager(geo=geo, **kwargs)
        else:
            self.augmenter = AugmentManager(**kwargs)

    def apply_augmenter(self, data: DataDict) -> DataDict:
        """Apply the configured augmenter, if present.

        Parameters
        ----------
        data : dict
            One sample dictionary produced by the dataset.

        Returns
        -------
        dict
            Augmented sample dictionary, or the input dictionary unchanged if
            no augmenter is configured.
        """
        if self.augmenter is None:
            return data

        return self.augmenter(data)

    @classmethod
    def metadata_dict(cls, data: DataDict) -> DataDict:
        """Extract standard dataset metadata from one reader output.

        Parameters
        ----------
        data : dict
            Raw sample dictionary returned by a reader.

        Returns
        -------
        dict
            Subset of ``data`` restricted to standard index and source
            provenance keys.
        """
        keep = set(cls._index_keys).union(cls._source_keys)
        return {key: data[key] for key in data if key in keep}

    @classmethod
    def index_data_types(cls) -> dict[str, str]:
        """Return the standard collate types for metadata keys.

        Returns
        -------
        dict[str, str]
            Mapping from standard metadata key name to the collate type used
            by :class:`spine.io.collate.CollateAll`.
        """
        keys = (*cls._index_keys, *cls._source_keys)
        return {key: "scalar" for key in keys}

    @classmethod
    def index_overlay_methods(cls) -> dict[str, str]:
        """Return the standard overlay methods for metadata keys.

        Returns
        -------
        dict[str, str]
            Mapping from standard metadata key name to the overlay method used
            by :class:`spine.io.overlay.Overlayer`.
        """
        keys = (*cls._index_keys, *cls._source_keys)
        return {key: "cat" for key in keys}
