"""Event access to aligned PIMM detector hits and label tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, ClassVar

import h5py
import numpy as np

from .base import ReaderBase

__all__ = ["PIMMReader"]


class PIMMReader(ReaderBase):
    """Delegate detector decoding and truth associations to ``pimm-data``.

    The upstream dataset joins hits and ``labl`` by task and event identity.
    SPINE reads its untransformed NumPy products, preserving particle-table
    row indexes until SPINE collation. No legacy pysupera file is read here.
    """

    name: ClassVar[str] = "pimm"

    # Public discovery options mirror the upstream dataset API.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        data_root: str = ".",
        files: Mapping[str, str | list[str]] | None = None,
        runs: str | list[str] | None = None,
        split: str | None = None,
        dataset_name: str = "sim_pixel",
        volume: int | None = None,
        keys: Sequence[str] | None = None,
        **selection: Any,
    ) -> None:
        """Index paired detector-hit and label shards.

        Parameters
        ----------
        data_root : str, default "."
            PIMM corpus root, containing modality or task directories.
        files : mapping, optional
            Explicit ``hits`` and ``labl`` paths or glob patterns.
        runs : str or list[str], optional
            Task directories or glob relative to the corpus root.
        split : str, optional
            Upstream split directory; defaults to flat discovery without runs.
        dataset_name : str, default "sim_pixel"
            Upstream file prefix.
        volume : int, optional
            Restrict input to one detector volume.
        keys : sequence[str], optional
            Only ``pimm`` is supported: it is one aligned upstream event.
        **selection : dict
            Standard ReaderBase entry selection. No run/subrun is invented.
        """
        if keys is not None and set(keys) != {"pimm"}:
            raise ValueError("PIMMReader exposes the aligned 'pimm' event product.")
        try:
            backend = import_module("pimm_data")
        except ImportError as err:
            raise ImportError(
                "PIMM input requires the optional spine[pimm] extra."
            ) from err

        # Keep all ownership references, including LE; semantic losses decide
        # which classes to supervise. Never apply upstream transforms twice.
        self.dataset = backend.JAXTPCDataset(
            data_root=data_root,
            files=files,
            runs=runs,
            split=("train" if runs is not None else "") if split is None else split,
            dataset_name=dataset_name,
            modalities=("hits",),
            labels="all",
            le_is_stuff=False,
            volume=volume,
            allow_partial=False,
        )
        reader = self.dataset.hits_reader
        self.file_paths = list(reader.h5_files)
        counts = np.array([len(index) for index in reader.indices])
        self.num_entries = int(counts.sum())
        self.file_offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
        self.file_index = np.repeat(np.arange(len(counts)), counts)
        self.process_entry_list(**selection)

        self.metadata = [
            self._read_metadata(path) for path in self.dataset.labl_reader.h5_files
        ]

    @staticmethod
    def _read_metadata(path: str) -> dict[str, Any]:
        """Read the small detector-geometry attributes from a label shard.

        Parameters
        ----------
        path : str
            PIMM labl shard path.

        Returns
        -------
        dict
            Pixel geometry and coordinate/value conventions.
        """
        with h5py.File(path, "r") as file:
            attrs: dict[str, Any] = dict(file["config"].attrs)
        geometry = attrs.get("supera_pixel_geometry")
        return {
            "pixel_geometry": json.loads(geometry) if geometry is not None else None,
            "hit_x_from": attrs.get("supera_hit_x_from"),
            "hit_energy": attrs.get("supera_hit_energy"),
        }

    def get(self, idx: int) -> dict[str, Any]:
        """Read one aligned event and its physical source identity.

        Parameters
        ----------
        idx : int
            Index in SPINE's selected entry list.

        Returns
        -------
        dict
            Nested ``pimm`` event and standard source provenance.
        """
        entry = int(self.entry_index[idx])
        file_index, event_num = self.dataset.hits_reader.locate(entry)
        labl_index, _ = self.dataset.labl_reader.locate(entry)
        event = self.dataset.get_data(entry)
        event["metadata"] = self.metadata[labl_index]
        identity = self.dataset.hits_reader.event_id_of(entry)
        event["identity"] = {"entry": event_num}
        if identity is not None:
            event["identity"]["event_id"] = identity
        result = {
            "index": idx,
            "file_index": file_index,
            "file_entry_index": event_num,
            "pimm": event,
        }
        result.update(self.get_source_provenance(file_index, event_num))
        return result

    def close(self) -> None:
        """Release upstream HDF5 handles before leaving a worker."""
        self.dataset.hits_reader.close()
        self.dataset.labl_reader.close()

    def __getstate__(self) -> dict[str, Any]:
        """Drop live handles before pickling for spawned loader workers."""
        self.close()
        return self.__dict__.copy()
