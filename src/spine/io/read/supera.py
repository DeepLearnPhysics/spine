"""Projected event reads from the pysupera 4.1 HDF5 format."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Any, ClassVar, cast

import h5py
import numpy as np

from .base import ReaderBase
from .hdf5.common import require_group

__all__ = ["SuperaReader"]


def _dataset(file: h5py.File | h5py.Group, key: str) -> h5py.Dataset:
    """Retrieve a required dataset and reject a group at the same path.

    Parameters
    ----------
    file : h5py.File or h5py.Group
        Open input file or parent group.
    key : str
        Required dataset path.

    Returns
    -------
    h5py.Dataset
        Dataset handle owned by the open file.

    Raises
    ------
    TypeError
        If the path refers to a group rather than a dataset.
    """
    value = file[key]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Expected a dataset at {key}.")
    return value


class SuperaReader(ReaderBase):
    """Read truth deposits and event-local particle ranges without loading hits.

    Products are ``points``, ``particles``, ``inter``, ``events``, ``metadata``
    and ``groups``. Hit labels require an explicit plane product such as
    ``hit_labels/volume0/plane0``. No HDF5 handles survive a read, making the
    reader safe to pickle or use in data-loader workers.
    """

    name: ClassVar[str] = "supera"
    default_keys = ("points", "particles", "inter", "events", "metadata")

    def __init__(
        self,
        file_keys: str | list[str] | None = None,
        file_list: str | None = None,
        keys: Sequence[str] | None = None,
        limit_num_files: int | None = None,
        max_print_files: int = 10,
        **selection: Any,
    ) -> None:
        """Resolve files, validate format and apply ReaderBase entry selection.

        Parameters
        ----------
        file_keys : str or list[str], optional
            Input paths or glob patterns.
        file_list : str, optional
            Text file containing input paths.
        keys : sequence[str], optional
            Raw products to read; defaults exclude groups and hit labels.
        limit_num_files : int, optional
            Maximum number of input files.
        max_print_files : int, default 10
            Maximum number of paths printed while resolving files.
        **selection : dict
            Entry selection arguments accepted by ReaderBase.process_entry_list.
            Run/event selection is unavailable because pysupera does not define
            run/subrun identities.
        """
        # Register optional compression filters used by upstream productions.
        try:
            import_module("hdf5plugin")
        except ImportError:
            pass

        # Inspect only format metadata and offsets while building the event index.
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        self.keys = tuple(self.default_keys if keys is None else keys)
        counts = [self._validate_file(path) for path in self.file_paths]

        # ReaderBase maps selected global entries back to their physical files.
        self.num_entries = sum(counts)
        self.file_offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
        self.file_index = np.repeat(np.arange(len(counts)), counts)
        self.process_entry_list(**selection)

    def _validate_file(self, path: str) -> int:
        """Check format metadata without materializing event payloads.

        Parameters
        ----------
        path : str
            Input HDF5 path.

        Returns
        -------
        int
            Number of events in the validated file.

        Raises
        ------
        TypeError
            If the format, offsets or required HDF5 object types are invalid.
        ValueError
            If the point layout or requested products are invalid.
        """
        # Pylint infers Group for h5py lookups even after _dataset validates them.
        # pylint: disable=no-member
        with h5py.File(path, "r") as file:
            if (
                "format_version" not in file
                or _dataset(file, "format_version").asstr()[()] != "4.1.0"
            ):
                raise TypeError(f"{path}: expected pysupera format 4.1.0.")

            # Each table has its own event offsets and row domain.
            num_entries = int(_dataset(file, "n_events")[()])
            for offset, target in (
                ("events/offsets", "particles/id"),
                ("points/offsets", "points/flat"),
                ("inter/offsets", "inter/interaction_id"),
            ):
                offsets = _dataset(file, offset)[:]
                if (
                    offsets.shape != (num_entries + 1,)
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) < 0)
                    or offsets[-1] != len(_dataset(file, target))
                ):
                    raise TypeError(f"{path}: invalid {offset}.")

            # Confirm column names and physical width before parsing point data.
            columns = tuple(_dataset(file, "points/columns").asstr()[:])
            if (
                columns[:3] != ("x", "y", "z")
                or "dE" not in columns
                or not {"t", "time"}.intersection(columns)
                or _dataset(file, "points/flat").ndim != 2
                or _dataset(file, "points/flat").shape[1] != len(columns)
            ):
                raise ValueError(f"{path}: unsupported point columns {columns}.")

            # Large hit-label arrays are opt-in through an explicit plane product.
            for key in self.keys:
                if key not in self.default_keys + ("groups",) and not (
                    key.startswith("hit_labels/")
                    and key.count("/") == 2
                    and key in file
                    and isinstance(file[key], h5py.Group)
                    and "offsets" in require_group(file, key)
                ):
                    raise ValueError(f"Unsupported Supera product: {key}")
        return num_entries

    def get(self, idx: int) -> dict[str, Any]:
        """Read selected products for one event, rebasing absolute ranges.

        Parameters
        ----------
        idx : int
            Index in the selected entry list.

        Returns
        -------
        dict[str, Any]
            Projected products and standard source-file provenance. Point and
            particle ranges are local to the returned event.
        """
        # Dataset members are validated by _dataset; h5py lookup inference is dynamic.
        # pylint: disable=no-member
        file_index = self.get_file_index(idx)
        entry_index = self.get_file_entry_index(idx)
        result: dict[str, Any] = {
            "index": idx,
            "file_index": file_index,
            "file_entry_index": entry_index,
        }
        result.update(self.get_source_provenance(file_index, entry_index))

        # Open per read so worker processes never inherit live HDF5 handles.
        with h5py.File(self.file_paths[file_index], "r") as file:
            point_start, point_stop = _dataset(file, "points/offsets")[
                entry_index : entry_index + 2
            ]

            # Hit labels require an explicit plane path; never default to all hits.
            for key in self.keys:
                if key == "metadata":
                    result[key] = dict(file.attrs)
                    result[key]["columns"] = tuple(
                        _dataset(file, "points/columns").asstr()[:]
                    )
                elif key == "points":
                    result[key] = _dataset(file, "points/flat")[point_start:point_stop]
                elif key == "events":
                    result[key] = {
                        name: _dataset(file, f"{key}/{name}")[entry_index]
                        for name in require_group(file, key)
                        if name != "offsets"
                    }
                else:
                    # Each physical table has its own event offsets.
                    result[key] = self._read_table(file, key, entry_index)
        return result

    @staticmethod
    def _read_table(
        file: h5py.File, key: str, entry_index: int
    ) -> dict[str, np.ndarray]:
        """Slice a columnar product and rebase its stored ranges.

        Parameters
        ----------
        file : h5py.File
            Open input file.
        key : str
            Particle, interaction, group or explicit hit-plane product path.
        entry_index : int
            Event index within this file.

        Returns
        -------
        dict[str, np.ndarray]
            Event-local columns. Hit and deposit-group labels retain their own
            row domains and are never assumed to align with points.
        """
        group = require_group(file, key)
        offset_key = "events/offsets" if key == "particles" else f"{key}/offsets"
        start, stop = _dataset(file, offset_key)[entry_index : entry_index + 2]
        point_start = _dataset(file, "points/offsets")[entry_index]
        particle_start = _dataset(file, "events/offsets")[entry_index]

        # Slice each column before performing any range arithmetic.
        table: dict[str, np.ndarray] = {}
        # HDF5 groups yield child names; their iterator typing is overly broad.
        for name in cast(Iterable[str], group):
            if name in ("offsets", "volume_offsets", "volume_offsets_index"):
                continue
            values = _dataset(group, name)[start:stop]
            if key in ("particles", "inter"):
                if "pc_" in name:
                    values = np.where(values >= 0, values - point_start, values)
                elif key == "inter" and name in ("part_start", "part_end"):
                    values = np.where(values >= 0, values - particle_start, values)
            table[name] = values

        # Volume starts index the event's deposit-group array, not its points.
        if key == "groups":
            start, stop = _dataset(group, "volume_offsets_index")[
                entry_index : entry_index + 2
            ]
            table["volume_offsets"] = _dataset(group, "volume_offsets")[start:stop]
        return table
