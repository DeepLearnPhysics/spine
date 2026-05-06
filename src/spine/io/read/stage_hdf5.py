"""Stage-aware HDF5 cache reader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from warnings import warn

import h5py
import numpy as np
import yaml
from yaml.parser import ParserError

from spine.utils.logger import logger

from .hdf5 import HDF5Reader

__all__ = ["StageHDF5Reader"]


class StageHDF5Reader(HDF5Reader):
    """Read products stored under one or more stage groups in a cache file.

    The reader exposes the same event-level interface as :class:`HDF5Reader`,
    but resolves requested product keys under ``/stages/<stage>`` instead of
    the flat top-level namespace.
    """

    name = "stage_hdf5"

    def __init__(
        self,
        stage: str | None = None,
        file_keys: str | list[str] | None = None,
        file_list: str | None = None,
        limit_num_files: int | None = None,
        max_print_files: int = 10,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: list[int] | None = None,
        skip_entry_list: list[int] | None = None,
        build_classes: bool = True,
        skip_unknown_attrs: bool = False,
        allow_missing: bool = False,
        keep_open: bool = True,
        swmr: bool = False,
        ignore_incomplete: bool = False,
        stage_map: Mapping[str, str] | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize the stage-cache reader.

        Parameters
        ----------
        stage : str, optional
            Default stage from which to load products. If omitted, keys are
            searched across all stages and must resolve uniquely.
        stage_map : mapping, optional
            Explicit map from product keys to stage names. This overrides the
            default stage on a per-product basis.
        keys : sequence[str], optional
            Product keys that should be exposed by the reader. If omitted, all
            products from the selected stage(s) are exposed.
        file_keys, file_list, limit_num_files, max_print_files, n_entry, n_skip, \
        entry_list, skip_entry_list, build_classes, skip_unknown_attrs, \
        allow_missing, keep_open, swmr, ignore_incomplete
            See :class:`spine.io.read.HDF5Reader`. These options control file
            discovery, entry selection, object reconstruction, file-handle
            lifetime, and incomplete-stage handling.
        """
        self.stage = stage
        self.stage_map = dict(stage_map or {})
        self.requested_keys = tuple(keys) if keys is not None else None
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        self.keep_open = keep_open
        self.swmr = swmr
        self.ignore_incomplete = ignore_incomplete
        self._handle_pid = None
        self._file_handles = {}
        self._resolved_products: dict[int, dict[str, str]] = {}
        self._source_info: dict[int, dict[str, object]] = {}

        file_index = []
        self.num_entries = 0
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as in_file:
                self._source_info[i] = self.read_source_info(in_file)
                product_stage_map = self.resolve_product_stages(in_file, path)
                self._resolved_products[i] = product_stage_map
                stage_lengths = self.get_stage_lengths(in_file, path, product_stage_map)
                num_entries = self.validate_stage_lengths(path, stage_lengths)
                file_index.append(i * np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        logger.info("Total number of entries in the file(s): %d\n", self.num_entries)

        self.file_index = (
            np.concatenate(file_index) if file_index else np.empty(0, dtype=np.int64)
        )
        self.run_info = None
        self.run_map = None

        self.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            None,
            None,
            allow_missing,
        )

        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs
        self.cfg = self.process_cfg()
        self.version = self.process_version()

    @staticmethod
    def get_stages_group(in_file: h5py.File, path: str) -> h5py.Group:
        """Return the top-level ``stages`` group.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            File path used to build informative error messages.
        """
        assert "stages" in in_file, f"Stage-cache file '{path}' is missing 'stages'."
        stages = in_file["stages"]
        assert isinstance(
            stages, h5py.Group
        ), f"'stages' in '{path}' must be a group, got {type(stages)}."
        return stages

    @classmethod
    def get_stage_group(cls, in_file: h5py.File, path: str, stage: str) -> h5py.Group:
        """Return one named stage group.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            File path used to build informative error messages.
        stage : str
            Name of the stage group to load under ``/stages``.
        """
        stages = cls.get_stages_group(in_file, path)
        assert (
            stage in stages
        ), f"Stage-cache file '{path}' does not contain stage '{stage}'."
        stage_group = stages[stage]
        assert isinstance(
            stage_group, h5py.Group
        ), f"Stage '{stage}' in '{path}' must be a group, got {type(stage_group)}."
        return stage_group

    @staticmethod
    def read_source_info(in_file: h5py.File) -> dict[str, object]:
        """Return top-level source provenance stored in the cache file.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.

        Returns
        -------
        dict[str, object]
            File-level provenance dictionary. If the cache predates the source
            group convention, this returns an empty dictionary.
        """
        if "source" not in in_file:
            return {}
        source_group = in_file["source"]
        assert isinstance(
            source_group, h5py.Group
        ), f"Expected 'source' to be a group, got {type(source_group)}."
        file_name = source_group.attrs["file_name"]
        if isinstance(file_name, bytes):
            file_name = file_name.decode()
        return {
            "source_file_name": file_name,
            "source_file_size": int(source_group.attrs["file_size"]),
            "source_file_mtime_ns": int(source_group.attrs["file_mtime_ns"]),
        }

    def list_stage_keys(self, stage_group: h5py.Group) -> tuple[str, ...]:
        """List product keys stored in one stage group.

        This excludes the administrative ``info`` and ``events`` members.
        """
        return tuple(key for key in stage_group.keys() if key not in {"info", "events"})

    def resolve_product_stages(self, in_file: h5py.File, path: str) -> dict[str, str]:
        """Resolve each requested product key to one stage.

        Resolution order is:

        1. explicit ``stage_map`` entry for the key
        2. dataset-level default ``stage``
        3. automatic discovery across all available stages

        Automatic discovery requires a unique match. If the same product name
        appears in multiple stages, the caller must disambiguate it.
        """
        if (
            self.stage is not None
            and self.requested_keys is None
            and not self.stage_map
        ):
            stage_group = self.get_stage_group(in_file, path, self.stage)
            self.check_stage_complete(stage_group, path, self.stage)
            return {key: self.stage for key in self.list_stage_keys(stage_group)}

        stages = self.get_stages_group(in_file, path)
        required_keys = (
            tuple(self.requested_keys)
            if self.requested_keys is not None
            else tuple(
                key
                for stage_name in stages
                for key in self.list_stage_keys(
                    self.get_stage_group(in_file, path, stage_name)
                )
            )
        )

        resolved: dict[str, str] = {}
        for key in required_keys:
            if key in self.stage_map:
                stage_name = self.stage_map[key]
                stage_group = self.get_stage_group(in_file, path, stage_name)
                self.check_stage_complete(stage_group, path, stage_name)
                if key not in stage_group:
                    raise KeyError(
                        f"Requested product '{key}' does not exist in stage "
                        f"'{stage_name}' of '{path}'."
                    )
                resolved[key] = stage_name
                continue

            if self.stage is not None:
                stage_group = self.get_stage_group(in_file, path, self.stage)
                self.check_stage_complete(stage_group, path, self.stage)
                if key not in stage_group:
                    raise KeyError(
                        f"Requested product '{key}' does not exist in stage "
                        f"'{self.stage}' of '{path}'."
                    )
                resolved[key] = self.stage
                continue

            candidates = []
            for stage_name in stages:
                stage_group = self.get_stage_group(in_file, path, stage_name)
                if key in stage_group:
                    self.check_stage_complete(stage_group, path, stage_name)
                    candidates.append(stage_name)

            if not candidates:
                raise KeyError(
                    f"Could not find requested product '{key}' in any stage of '{path}'."
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Requested product '{key}' appears in multiple stages of '{path}': "
                    f"{candidates}. Specify its stage explicitly."
                )

            resolved[key] = candidates[0]

        return resolved

    def check_stage_complete(
        self, stage_group: h5py.Group, path: str, stage: str
    ) -> None:
        """Reject incomplete stages unless explicitly allowed.

        Parameters
        ----------
        stage_group : h5py.Group
            Resolved stage group.
        path : str
            Cache file path.
        stage : str
            Stage name used in the error message.
        """
        if (
            "info" in stage_group
            and "complete" in stage_group["info"].attrs
            and not stage_group["info"].attrs["complete"]
            and not self.ignore_incomplete
        ):
            raise RuntimeError(
                f"Stage '{stage}' in '{path}' is marked incomplete. "
                "Pass ignore_incomplete=True to override."
            )

    def get_stage_lengths(
        self, in_file: h5py.File, path: str, product_stage_map: Mapping[str, str]
    ) -> dict[str, int]:
        """Return the event count of each referenced stage.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            Cache file path.
        product_stage_map : mapping
            Mapping from requested raw product key to resolved stage name.
        """
        stage_lengths: dict[str, int] = {}
        for stage_name in set(product_stage_map.values()):
            stage_group = self.get_stage_group(in_file, path, stage_name)
            events = stage_group["events"]
            assert isinstance(
                events, h5py.Dataset
            ), f"Stage '{stage_name}' in '{path}' is missing an 'events' dataset."
            stage_lengths[stage_name] = len(events)
        return stage_lengths

    @staticmethod
    def validate_stage_lengths(path: str, stage_lengths: Mapping[str, int]) -> int:
        """Ensure all referenced stages in one file have the same length.

        Returns
        -------
        int
            Shared number of entries across all referenced stages.
        """
        lengths = list(stage_lengths.values())
        if not lengths:
            return 0
        if any(length != lengths[0] for length in lengths[1:]):
            raise ValueError(
                f"Referenced stages in '{path}' do not expose the same number of entries: "
                f"{dict(stage_lengths)}."
            )
        return lengths[0]

    def process_cfg(self) -> dict[str, object] | None:
        """Return the stored configuration for the referenced stage(s), if any.

        Returns
        -------
        dict or object or None
            Parsed YAML configuration stored under stage metadata. A single
            stage yields its parsed object directly; multiple stages return a
            mapping from stage name to parsed object.
        """
        with h5py.File(self.file_paths[0], "r") as in_file:
            stage_names = sorted(set(self._resolved_products[0].values()))
            cfg_map: dict[str, object | None] = {}
            for stage_name in stage_names:
                stage_group = self.get_stage_group(
                    in_file, self.file_paths[0], stage_name
                )
                if "info" not in stage_group or "cfg" not in stage_group["info"].attrs:
                    cfg_map[stage_name] = None
                    continue
                cfg_str = stage_group["info"].attrs["cfg"]
                try:
                    assert isinstance(cfg_str, str), "'cfg' attribute is not a string."
                    cfg_map[stage_name] = yaml.safe_load(cfg_str)
                except ParserError:
                    warn(
                        "Parsing stage configuration failed, returning None for "
                        f"stage '{stage_name}'."
                    )
                    cfg_map[stage_name] = None

        if len(cfg_map) == 1:
            return next(iter(cfg_map.values()))
        return cfg_map

    def get(self, idx: int) -> dict[str, object]:
        """Return one merged cache entry.

        Parameters
        ----------
        idx : int
            Dataset entry index in the staged cache.

        Returns
        -------
        dict[str, object]
            Raw merged event dictionary containing standard metadata plus all
            requested stage products for the selected entry.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}."
            )
        file_idx = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        data: dict[str, object] = {
            "file_index": file_idx,
            "file_entry_index": entry_idx,
        }
        data.update(self._source_info.get(file_idx, {}))
        product_stage_map = self._resolved_products[file_idx]

        in_file, should_close = self._open_file(file_idx)
        try:
            for stage_name in sorted(set(product_stage_map.values())):
                stage_group = self.get_stage_group(
                    in_file, self.file_paths[file_idx], stage_name
                )
                events = stage_group["events"]
                assert isinstance(
                    events, h5py.Dataset
                ), f"Stage '{stage_name}' is missing an 'events' dataset."
                event = events[entry_idx]
                names = getattr(getattr(event, "dtype", None), "names", None)
                if names is None:
                    raise ValueError(
                        f"Stage '{stage_name}' event entry does not have named fields."
                    )
                for key in names:
                    if product_stage_map.get(key) != stage_name:
                        continue
                    self.load_key(stage_group, event, data, key)
        finally:
            if should_close:
                in_file.close()

        data["index"] = idx
        return data
