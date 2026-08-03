"""Module to write the output of the reconstruction to file."""

import os
from typing import Any

import h5py
import numpy as np
import yaml

import spine.data
from spine.version import __version__

from .common import DataFormat, decode_string_attribute, require_group
from .product import ProductGroupBackend
from .region import RegionReferenceBackend
from .schema import SchemaDiscoveryBackend

__all__ = ["HDF5Writer"]


class HDF5Writer(
    ProductGroupBackend,
    RegionReferenceBackend,
    SchemaDiscoveryBackend,
):
    """Write reconstruction data using a versioned SPINE HDF5 layout.

    Builds an HDF5 file to store the input and/or the output of the
    reconstruction chain. It can also be used to append an existing HDF5 file
    with information coming out of the analysis tools.

    The writer separates the SPINE software version from the physical storage
    version. ``spine_version`` records the producing release, while
    ``format_version`` selects one of two on-disk layouts:

    - V1 stores event-level HDF5 region references and uses HDF5 VLEN dtypes
      for variable object attributes.
    - V2 replaces those references and VLEN fields with flat datasets and
      integer offsets. It appends complete batches collectively to reduce
      dataset-resize and small-write overhead.

    V1 remains the default during the V2 rollout. Readers auto-detect both
    layouts, but a writer must use one layout consistently for the lifetime of
    a file.

    Typical configuration should look like:

    .. code-block:: yaml

        io:
          ...
          writer:
            name: hdf5
            file_name: output.h5
            keys:
              - input_data
              - segmentation
              - ...
    """

    name = "hdf5"
    DataFormat = DataFormat
    # `format_name` identifies this family of files. The integer version below
    # identifies its physical schema and is intentionally independent of the
    # package release in `spine.version.__version__`.
    format_name = "spine_hdf5"
    legacy_format_version = 1
    current_format_version = 2
    supported_format_versions = (legacy_format_version, current_format_version)
    source_index_keys = {
        "file_index": "source_file_index",
        "file_entry_index": "source_file_entry_index",
    }

    def __init__(
        self,
        file_name: str | None = None,
        directory: str | None = None,
        prefix: str | list[str] | None = None,
        suffix: str = "spine",
        keys: list[str] | None = None,
        skip_keys: list[str] | None = None,
        dummy_ds: dict[str, str] | None = None,
        overwrite: bool = False,
        append: bool = False,
        split: bool = False,
        lite: bool = False,
        keep_open: bool = True,
        flush_frequency: int | None = None,
        format_version: int = legacy_format_version,
    ) -> None:
        """Initializes the basics of the output file.

        Parameters
        ----------
        file_name : str, optional
            Name of the output HDF5 file
        directory : str, optional
            Output directory. When provided, all generated file names are
            relocated into this directory while preserving their resolved base
            names.
        prefix : str or List[str], optional
            Input file prefix. It will be use to form the output file name,
            provided that no file_name is explicitly provided. Must be a list
            with one prefix per input file when `split` is `True`.
        suffix : str, default "spine"
            Suffix to add to the output file name if it is built from the input
        keys : List[str], optional
            List of data product keys to store. If not specified, store everything
        skip_keys: List[str], optionl
            List of data product keys to skip
        dummy_ds: Dict[str, str], optional
            Keys for which to create placeholder datasets. For each key, specify
            the object type it is supposed to represent as a string.
        overwrite : bool, default False
            If `True`, overwrite the output file if it already exists
        append : bool, default False
            If `True`, add new values to the end of an existing file
        split : bool, default False
            If `True`, split the output to produce one file per input file
        lite : bool, default False
            If `True`, the lite version of objects is stored (drop point indexes)
        keep_open : bool, default True
            If `True`, keep one append handle open per output file and per
            process. This reduces HDF5 open/close churn when writing many
            batches. If `False`, open and close the file on each write call.
        flush_frequency : int, optional
            If specified, flush each output file after this many appended
            entries. If `None`, only flush when explicitly requested or when
            the file handle is closed.
        format_version : int, default 1
            Physical HDF5 layout version. Version 1 is the legacy
            region-reference/VLEN layout. Version 2 stores event and object
            boundaries as integer offsets and variable object attributes in
            flat datasets. The choice is persisted in
            ``info.attrs["format_version"]`` and cannot change when appending.
        """
        # Build the output file name(s) from the input prefix(es) if not provided
        self.file_names = self.get_file_names(
            file_name, prefix, suffix, split, directory
        )

        # Check that the output file(s) do(es) not already exist, if requested
        if not overwrite and not append:
            for file_name in self.file_names:
                if os.path.isfile(file_name):
                    raise FileExistsError(f"File with name {file_name} already exists.")
        elif overwrite and not append:
            for file_name in self.file_names:
                if os.path.isfile(file_name):
                    os.remove(file_name)

        # Store other persistent attributes
        self.append = append
        self.split = split
        self.lite = lite
        self.keep_open = keep_open
        self.flush_frequency = flush_frequency
        if format_version not in self.supported_format_versions:
            raise ValueError(
                f"Unsupported HDF5 format version {format_version}. Supported "
                f"versions are {self.supported_format_versions}."
            )
        self.format_version = format_version

        self.keys = set(keys) if keys is not None else None
        self.skip_keys = skip_keys

        # Initialize dummy dataset placeholders once
        self.dummy_ds = dummy_ds
        if self.dummy_ds is not None:
            for key, class_name in self.dummy_ds.items():
                self.dummy_ds[key] = getattr(spine.data, class_name)()

        # Initialize attributes to be stored when the output file is created
        self.ready = False
        self.object_dtypes = []
        self.type_dict = None
        self.product_metadata: dict[str, dict[str, Any]] = {}
        self.product_children: dict[str, tuple[str, str]] = {}
        self.event_dtype = None
        self._handle_pid: int | None = None
        self._file_handles: dict[int, h5py.File] = {}
        self._cfg: dict[str, Any] | None = None
        self._initialized_file_ids: set[int] = set()
        self._completed_file_ids: set[int] = set()
        self._entries_since_flush_by_file_id: dict[int, int] = {}
        self._max_written_file_id: int | None = None
        self._split_sequential = True

    def __enter__(self) -> "HDF5Writer":
        """Return the writer for use in a `with` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close persistent output handles on context-manager exit.

        Parameters
        ----------
        exc_type : type, optional
            Exception type raised inside the context, if any
        exc_val : Exception, optional
            Exception instance raised inside the context, if any
        exc_tb : traceback, optional
            Traceback associated with the raised exception, if any

        Returns
        -------
        bool
            Always `False` so exceptions are propagated to the caller.
        """
        if exc_type is None:
            self.finalize()
        self.close()
        return False

    def close(self) -> None:
        """Close any persistent HDF5 output handles owned by this writer.

        This only affects handles cached in the current process. It is safe to
        call repeatedly.
        """
        for handle in getattr(self, "_file_handles", {}).values():
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass

        self._file_handles = {}
        self._handle_pid = None

    def flush(self) -> None:
        """Flush all persistent HDF5 output handles to disk.

        This is useful when the writer keeps files open for a long time and the
        caller wants to force buffered metadata and dataset updates to disk.
        """
        for handle in self._file_handles.values():
            handle.flush()

    def finalize(self) -> None:
        """Mark initialized output files as complete and flush metadata.

        This method should only be called once the caller knows writing
        completed successfully for the relevant files.
        """
        for file_id in sorted(self._initialized_file_ids - self._completed_file_ids):
            if self.keep_open:
                handle, _ = self._get_output_handle(file_id)
                handle["info"].attrs["complete"] = True
                handle.flush()
            else:
                with h5py.File(self.file_names[file_id], "a") as out_file:
                    out_file["info"].attrs["complete"] = True

            self._completed_file_ids.add(file_id)

    def __del__(self) -> None:
        """Best-effort cleanup of persistent output handles on teardown."""
        self.close()

    def _check_handle_pid(self) -> None:
        """Ensure persistent output handles are only used in one process.

        Writer instances are not safe to share across processes. Unlike the
        reader, the writer refuses PID changes outright to avoid ambiguous
        multi-process append behavior.
        """
        current_pid = os.getpid()
        if self._handle_pid is None:
            self._handle_pid = current_pid
            return

        if self._handle_pid != current_pid:
            raise RuntimeError(
                "HDF5Writer file handles are process-local and cannot be reused "
                "across process boundaries."
            )

    def _get_output_handle(self, file_id: int) -> tuple[h5py.File, bool]:
        """Return an appendable HDF5 handle for one output file.

        Parameters
        ----------
        file_id : int
            Position of the target file in `self.file_names`

        Returns
        -------
        tuple[h5py.File, bool]
            The opened HDF5 file handle and a flag indicating whether the
            caller is responsible for closing it. The flag is `True` only when
            `keep_open=False`.
        """
        self._ensure_file(file_id)
        if not self.keep_open:
            return h5py.File(self.file_names[file_id], "a"), True

        self._check_handle_pid()
        handle = self._file_handles.get(file_id)
        if handle is None or not handle.id.valid:
            handle = h5py.File(self.file_names[file_id], "a")
            self._file_handles[file_id] = handle

        return handle, False

    def _ensure_file(self, file_id: int) -> None:
        """Create or prepare one output file for writing on first use."""
        if file_id in self._completed_file_ids:
            raise RuntimeError(
                f"Output file '{self.file_names[file_id]}' was already finalized."
            )

        if file_id in self._initialized_file_ids:
            return

        file_name = self.file_names[file_id]
        file_exists = os.path.isfile(file_name)
        if self.append and file_exists:
            if self.keep_open:
                self._check_handle_pid()
                out_file = h5py.File(file_name, "a")
                self._file_handles[file_id] = out_file
                self._validate_append_format(out_file, file_name)
                event_obj = out_file["events"]
                assert isinstance(event_obj, h5py.Dataset), (
                    "Expected dataset for events to be a Dataset, but got "
                    f"{type(event_obj)} instead."
                )
                self.event_dtype = getattr(event_obj, "dtype")
                out_file["info"].attrs["complete"] = False
            else:
                with h5py.File(file_name, "a") as out_file:
                    self._validate_append_format(out_file, file_name)
                    event_obj = out_file["events"]
                    assert isinstance(event_obj, h5py.Dataset), (
                        "Expected dataset for events to be a Dataset, but got "
                        f"{type(event_obj)} instead."
                    )
                    self.event_dtype = getattr(event_obj, "dtype")
                    out_file["info"].attrs["complete"] = False
        else:
            self._ensure_parent_dir(file_name)
            if self.keep_open:
                self._check_handle_pid()
                out_file = h5py.File(file_name, "w")
                self._file_handles[file_id] = out_file
            else:
                out_file = h5py.File(file_name, "w")

            try:
                out_file.create_group("info")
                # Keep the historical `version` attribute for old consumers,
                # while giving software and physical layout explicit names.
                out_file["info"].attrs["version"] = __version__
                out_file["info"].attrs["spine_version"] = __version__
                out_file["info"].attrs["format"] = self.format_name
                out_file["info"].attrs["format_version"] = self.format_version
                out_file["info"].attrs["complete"] = False
                if self._cfg is not None:
                    out_file["info"].attrs["cfg"] = yaml.dump(self._cfg)
                assert (
                    self.type_dict is not None
                ), "Cannot initialize an output file before data types are known."
                if self.format_version == self.legacy_format_version:
                    self.initialize_region_datasets(out_file, self.type_dict)
                else:
                    self.initialize_product_datasets(out_file, self.type_dict)
            finally:
                if not self.keep_open:
                    out_file.close()

        self._initialized_file_ids.add(file_id)
        self._entries_since_flush_by_file_id[file_id] = 0

    def _validate_append_format(self, out_file: h5py.File, file_name: str) -> None:
        """Ensure an existing output file uses the requested physical layout.

        Mixing layouts in one file would invalidate all event-boundary
        assumptions: V1 events contain region references, whereas V2 product
        groups maintain independent offset arrays. Files without explicit
        layout metadata predate V2 and are therefore treated as V1.

        Parameters
        ----------
        out_file : h5py.File
            Existing file opened for append.
        file_name : str
            File name included in validation errors.

        Raises
        ------
        ValueError
            If metadata is missing or the stored and requested versions differ.
        """
        if "info" not in out_file:
            raise ValueError(f"Cannot append to '{file_name}': missing info group.")
        stored_version = int(
            out_file["info"].attrs.get("format_version", self.legacy_format_version)
        )
        if stored_version != self.format_version:
            raise ValueError(
                f"Cannot append HDF5 format version {self.format_version} to "
                f"'{file_name}', which uses format version {stored_version}."
            )
        if stored_version == self.current_format_version:
            if "products" not in out_file:
                raise ValueError(
                    f"Cannot append to '{file_name}': missing V2 products group."
                )
            products = require_group(out_file, "products")
            for key, metadata in self.product_metadata.items():
                if key not in products:
                    raise ValueError(
                        f"Cannot append product `{key}` to '{file_name}': "
                        "stored product metadata is missing."
                    )
                product = require_group(products, key)
                if "product_metadata" not in product.attrs:
                    raise ValueError(
                        f"Cannot append product `{key}` to '{file_name}': "
                        "stored product metadata is missing."
                    )
                encoded = decode_string_attribute(
                    product.attrs["product_metadata"], "product_metadata"
                )
                stored = yaml.safe_load(encoded)
                if stored != metadata:
                    raise ValueError(
                        f"Cannot append product `{key}` to '{file_name}': "
                        "the stored and incoming schemas differ."
                    )

            for parent, name in self.product_children.values():
                if parent not in products:
                    raise ValueError(
                        f"Cannot append product `{parent}` to '{file_name}': "
                        f"stored child `{name}` is missing."
                    )
                parent_group = require_group(products, parent)
                if name not in parent_group:
                    raise ValueError(
                        f"Cannot append product `{parent}` to '{file_name}': "
                        f"stored child `{name}` is missing."
                    )

    @staticmethod
    def _ensure_parent_dir(file_name: str) -> None:
        """Create the parent directory for an output file, if needed."""
        dir_name = os.path.dirname(file_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def _record_write(self, file_id: int, count: int, out_file: h5py.File) -> None:
        """Update flush bookkeeping for one file after appending entries."""
        if self.flush_frequency is None:
            return

        self._entries_since_flush_by_file_id[file_id] += count
        if self._entries_since_flush_by_file_id[file_id] >= self.flush_frequency:
            out_file.flush()
            self._entries_since_flush_by_file_id[file_id] = 0

    def _finalize_split_predecessors(self, current_file_ids: np.ndarray) -> None:
        """Finalize older split outputs once the writer advances monotonically."""
        if (
            not self.split
            or self._max_written_file_id is None
            or len(current_file_ids) == 0
        ):
            return

        min_file_id = int(np.min(current_file_ids))
        if min_file_id < self._max_written_file_id:
            self._split_sequential = False
            return

        if not self._split_sequential or min_file_id <= self._max_written_file_id:
            return

        for file_id in sorted(self._initialized_file_ids - self._completed_file_ids):
            if file_id < min_file_id:
                if self.keep_open:
                    handle, _ = self._get_output_handle(file_id)
                    handle["info"].attrs["complete"] = True
                    handle.flush()
                else:
                    with h5py.File(self.file_names[file_id], "a") as out_file:
                        out_file["info"].attrs["complete"] = True
                self._completed_file_ids.add(file_id)

    @staticmethod
    def get_file_names(
        file_name: str | None = None,
        prefix: str | list[str] | None = None,
        suffix: str = "spine",
        split: bool = False,
        directory: str | None = None,
    ) -> list[str]:
        """Build output file name(s) from an explicit name or input prefix(es).

        Logic is as follows:

        - If `split` is `False` and `file_name` is provided, use `file_name`
        - If `split` is `False` and `file_name` is not provided, build the file name
          from the input `prefix` by adding a suffix
        - If `split` is `True` and `file_name` is not provided, build the file names
          from the input `prefix` by adding a suffix
        - If `split` is `True` and `file_name` is provided, build the file names from
          `file_name` by adding an index, unless there is only one input prefix,
          in which case use `file_name` as is

        Parameters
        ----------
        file_name : str, optional
            Name of the output HDF5 file. If not provided, it will be built from the
            input prefix(es).
        prefix : str or List[str], optional
            Input file prefix(es).
        suffix : str, default "spine"
            Suffix to add to the output file name if it is built from the input
        split : bool, default False
            If `True`, split the output to produce one file per input file.
        directory : str, optional
            Output directory. When provided, the resolved output file base name
            is placed under this directory regardless of the directory encoded
            in ``file_name`` or ``prefix``.

        Returns
        -------
        List[str]
            List of output file names.
        """

        def relocate(path: str) -> str:
            """Move one resolved output file name into the requested directory."""
            if directory is None:
                return path
            return os.path.join(directory, os.path.basename(path))

        # If the output is not split, use the provided file name or build it from the prefix
        if not split:
            if file_name:
                return [relocate(file_name)]

            assert prefix is not None and isinstance(prefix, str), (
                "If the output `file_name` is not provided, must provide "
                "the input file `prefix` to build it from."
            )
            prefix_dir = directory if directory is not None else os.path.dirname(prefix)
            prefix_base = os.path.splitext(os.path.basename(prefix))[0]
            return [os.path.join(prefix_dir, f"{prefix_base}_{suffix}.h5")]

        # If the output is split, build the file names from the provided one by
        # adding an index, unless there is only one prefix per file,
        # in which case use the provided name as is
        assert prefix is not None and not isinstance(prefix, str), (
            "If `split` is enabled, must provide one `prefix` per input file "
            "to determine the number of output files."
        )

        if file_name and len(prefix) == 1:
            return [relocate(file_name)]

        if not file_name:
            output_dir = (
                directory if directory is not None else os.path.dirname(prefix[0])
            )
            return [
                os.path.join(
                    output_dir,
                    f"{os.path.splitext(os.path.basename(pre))[0]}_{suffix}.h5",
                )
                for pre in prefix
            ]

        # Otherwise, build the file names from the provided one by adding an index
        dir_name = directory if directory is not None else os.path.dirname(file_name)
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        return [
            os.path.join(dir_name, f"{base_name}_{i}.h5") for i in range(len(prefix))
        ]

    def create(
        self,
        data: dict[str, Any],
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the output file structure based on the data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data products
        cfg : Dict[str, Any]
            Dictionary containing the complete SPINE configuration
        """
        # Fetch the required keys to be stored and register them
        self.keys = self.get_stored_keys(data)
        self._cfg = cfg

        # Fetch the data type information for each key and store it in a dictionary
        self.type_dict, self.object_dtypes = self.get_data_formats(data, self.keys)

        # Mark file(s) as ready for use
        self.ready = True

    def with_source_provenance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a data dictionary augmented with persisted source provenance.

        When upstream products carry `file_index` and/or `file_entry_index`,
        preserve those values under explicit `source_*` names so they survive
        round-tripping through HDF5 without colliding with the reader-owned
        runtime index fields of the produced HDF5 file.

        Parameters
        ----------
        data : dict
            Dictionary of data products to be written

        Returns
        -------
        dict
            Shallow copy of the data dictionary with `source_*` aliases added
            when the corresponding upstream index fields are present.
        """
        aliased = dict(data)
        for key, source_key in self.source_index_keys.items():
            if key in aliased and source_key not in aliased:
                aliased[source_key] = aliased[key]

        return aliased

    def expand_cluster_labels(self, data: dict[str, Any]) -> dict[str, Any]:
        """Expand structured cluster labels into generic HDF5 products.

        A compact voxel table remains under the original key. Its optional
        named particle table is stored under ``<key>_particles`` as a compound
        array. The HDF5 cluster-label parser fuses these products again when
        reading a cache.

        Parameters
        ----------
        data : dict
            Batched writer input containing zero or more structured cluster
            label products.

        Returns
        -------
        dict
            Shallow copy with cluster-label products lowered to serializable
            voxel arrays and optional particle sidecars.
        """
        expanded = dict(data)
        for key, value in tuple(data.items()):
            # Normalize batched and already-unwrapped structured products
            if isinstance(value, spine.data.ClusterLabelBatch):
                entries = [value[i] for i in range(value.batch_size)]
            elif (
                isinstance(value, list)
                and len(value)
                and isinstance(value[0], spine.data.ClusterLabelData)
            ):
                entries = value
            else:
                continue

            # Keep the compact voxel association table under the public key
            expanded[key] = [entry.data for entry in entries]
            if entries[0].particles is None:
                continue

            # Materialize named fields as serializable particle records
            particle_key = f"{key}_particles"
            particle_arrays = self._serialize_particle_tables(entries)

            # Store the sidecar even when the writer is restricted to selected keys
            expanded[particle_key] = particle_arrays
            if self.keys is not None and key in self.keys:
                self.keys.add(particle_key)

        return expanded

    def __call__(self, data: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
        """Append the HDF5 file with the content of a batch.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        cfg : dict
            Dictionary containing the complete SPINE configuration
        """
        # Preserve the original source provenance under explicit names.
        data = self.with_source_provenance(data)
        if self.format_version == self.current_format_version:
            data = self.prepare_products(data)
        else:
            data = self.expand_cluster_labels(data)

        # Nest data if is not already, fetch batch size
        if np.isscalar(data["index"]):
            for k in data:
                data[k] = [data[k]]
            batch_size = 1
        else:
            batch_size = len(data["index"])

        # If needed, add empty data for dummy datasets
        if self.dummy_ds is not None:
            for key, value in self.dummy_ds.items():
                data[key] = [spine.data.ObjectList([], default=value)] * batch_size

        # If this function has never been called, initialiaze the HDF5 file(s)
        if not self.ready:
            self.create(data, cfg)

        # Append file(s). V1 preserves its entry-at-a-time path for backward
        # compatibility. V2 handles the complete batch together so each flat
        # values/offset dataset is resized at most once per product and batch.
        if not self.split or len(self.file_names) == 1:
            out_file, should_close = self._get_output_handle(0)
            try:
                batch_ids = np.arange(batch_size, dtype=np.int64)
                if self.format_version == self.current_format_version:
                    self.append_product_entries(out_file, data, batch_ids)
                else:
                    for batch_id in batch_ids:
                        self.append_region_entry(out_file, data, int(batch_id))
                self._record_write(0, batch_size, out_file)
            finally:
                if should_close:
                    out_file.close()

        else:
            file_ids = np.asarray(data["file_index"], dtype=np.int64)
            unique_file_ids = np.unique(file_ids)
            self._finalize_split_predecessors(unique_file_ids)
            for file_id in np.unique(file_ids):
                out_file, should_close = self._get_output_handle(int(file_id))
                try:
                    batch_ids = np.where(file_ids == file_id)[0]
                    if self.format_version == self.current_format_version:
                        self.append_product_entries(out_file, data, batch_ids)
                    else:
                        for batch_id in batch_ids:
                            self.append_region_entry(out_file, data, int(batch_id))
                    self._record_write(int(file_id), len(batch_ids), out_file)
                finally:
                    if should_close:
                        out_file.close()

            max_file_id = int(np.max(unique_file_ids))
            if (
                self._max_written_file_id is None
                or max_file_id > self._max_written_file_id
            ):
                self._max_written_file_id = max_file_id
