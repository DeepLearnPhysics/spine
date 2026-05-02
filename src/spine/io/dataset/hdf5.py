"""Dataset wrapper around :class:`spine.io.read.HDF5Reader`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from spine.utils.conditional import TORCH_AVAILABLE

from ..read import HDF5Reader
from .base import BaseDataset, DataDict

__all__ = ["HDF5Dataset"]


class HDF5Dataset(BaseDataset):
    """Thin torch dataset wrapper around :class:`HDF5Reader`."""

    name: ClassVar[str] = "hdf5"
    reader: HDF5Reader

    def __init__(
        self,
        dtype: str | None = None,
        keys: Sequence[str] | None = None,
        skip_keys: Sequence[str] | None = None,
        data_types: Mapping[str, str] | None = None,
        overlay_methods: Mapping[str, str] | None = None,
        augment: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the HDF5-backed dataset."""
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use HDF5Dataset.")
        if keys is not None and skip_keys is not None:
            raise ValueError("Provide either `keys` or `skip_keys`, not both.")

        self.keys = set(keys) if keys is not None else None
        self.skip_keys = set(skip_keys) if skip_keys is not None else set()
        self._data_types = dict(data_types) if data_types is not None else None
        self._overlay_methods = (
            dict(overlay_methods) if overlay_methods is not None else None
        )

        _ = dtype  # Accepted for factory compatibility.
        self.build_augmenter(augment)
        self.reader = HDF5Reader(**kwargs)

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self.reader)

    def __getitem__(self, idx: int) -> DataDict:
        """Return one cached dataset entry."""
        result = self.reader[idx]
        if self.keys is not None:
            keep = self.keys.union(self._index_keys)
            result = {key: val for key, val in result.items() if key in keep}

        for key in self.skip_keys:
            result.pop(key, None)

        return self.apply_augmenter(result)

    @property
    def data_types(self) -> dict[str, str]:
        """Return the collate type for each HDF5 product."""
        data_types = self.index_data_types()
        if self._data_types is not None:
            data_types.update(self._data_types)
        else:
            sample = self[0] if len(self) else {}
            for key in sample:
                if key not in data_types:
                    data_types[key] = "list"

        return data_types

    @property
    def overlay_methods(self) -> dict[str, str]:
        """Return the overlay method for each HDF5 product."""
        overlay_methods = self.index_overlay_methods()
        if self._overlay_methods is not None:
            overlay_methods.update(self._overlay_methods)

        return overlay_methods

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return the names of all data products exposed by the dataset."""
        return tuple(self.data_types.keys())
