"""Shared helpers for torch-backed datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from spine.utils.conditional import TORCH_AVAILABLE

from ..augment import AugmentManager

if TORCH_AVAILABLE:
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Import-safe stand-in used when PyTorch is unavailable."""


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

    def __getitems__(self, indices: Sequence[Any]) -> list[DataDict]:
        """Return a batch of samples using the scalar access fallback.

        PyTorch's map-style data loader uses this optional plural method when
        it is available. Concrete datasets can override it to combine backend
        reads without changing the samples presented to the collate function.

        Parameters
        ----------
        indices : sequence
            Dataset indexes requested by the data loader.

        Returns
        -------
        list[dict]
            Samples in the same order as ``indices``.
        """
        return [self[index] for index in indices]

    @staticmethod
    def load_batch(dataset: Any, indices: Sequence[Any]) -> list[DataDict]:
        """Load multiple samples from a child dataset.

        The optimized plural interface is preferred when available. Plain
        map-style datasets remain supported through scalar indexing, which is
        useful for composite datasets configured with third-party sources.

        Parameters
        ----------
        dataset : object
            Child dataset to read.
        indices : sequence
            Indexes to forward to the child dataset.

        Returns
        -------
        list[dict]
            Child samples in the requested order.
        """
        getitems = getattr(dataset, "__getitems__", None)
        if getitems is not None:
            return getitems(indices)

        return [dataset[index] for index in indices]

    def build_augmenter(self, augment: Mapping[str, Any] | None) -> None:
        """Instantiate the configured augmenter, if any.

        Parameters
        ----------
        augment : mapping, optional
            Augmentation configuration block passed to
            :class:`spine.io.augment.AugmentManager`.
        """
        if augment is None:
            self.augmenter = None
            return

        kwargs = dict(augment)
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
