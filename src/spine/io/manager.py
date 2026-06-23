"""Input/output manager used by the central SPINE driver."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.stopwatch import StopwatchManager

from .factories import loader_factory, reader_factory, writer_factory
from .unwrap import Unwrapper

__all__ = ["IOManager"]


class IOManager:
    """Central I/O setup manager.

    The manager owns the I/O-specific setup rules used by
    :class:`spine.driver.Driver`. It selects exactly one input path, either a
    PyTorch-style loader or an entry reader, exposes the underlying reader used
    for event addressing, derives log/output prefixes from input file names,
    configures the optional writer and harmonizes ``iterations`` and ``epochs``.
    It also owns loader-batch unwrapping because that operation converts the
    loader I/O view into the per-entry view consumed by writers, builders,
    post-processors and analysis scripts.

    Attributes
    ----------
    loader : object or None
        Instantiated PyTorch DataLoader-like object. ``None`` when running from
        a reader.
    loader_iter : iterator or None
        Loader iterator used for sequential batch access.
    watch : StopwatchManager
        Stopwatch manager for I/O-owned operations. The driver may merge this
        into its aggregate stopwatch manager after processing.
    unwrapper : Unwrapper or None
        Batch unwrapper initialized when requested for loader-based execution.
        This belongs to the I/O boundary: reader mode already yields one entry,
        while loader mode may need to convert batched values into per-entry
        value lists.
    reader : object
        Reader object backing either the configured reader or the loader
        dataset.
    writer : object or None
        Optional writer object used to persist driver outputs.
    iter_per_epoch : int
        Number of batches or entries in one pass over the input.
    post_list : tuple[str, ...] or None
        Names of post-processors already recorded in the input file, if
        available from the reader metadata.
    log_prefix : str
        Compact input-derived prefix used for log file names.
    output_prefix : str or list[str]
        Output prefix passed to writers. This is a list when ``split_output``
        is enabled.
    iterations : int or None
        Number of driver iterations to run after harmonizing with ``epochs``.
    epochs : float or None
        Number of epochs to run after harmonizing with ``iterations``.
    """

    def __init__(
        self,
        loader: Mapping[str, Any] | None = None,
        reader: Mapping[str, Any] | None = None,
        writer: Mapping[str, Any] | None = None,
        *,
        rank: int | None = None,
        dtype: str = "float32",
        world_size: int = 0,
        distributed: bool = False,
        unwrap: bool = False,
        iterations: int | None = None,
        epochs: float | None = None,
        split_output: bool = False,
    ) -> None:
        """Initialize the I/O manager.

        Parameters
        ----------
        loader : mapping, optional
            PyTorch DataLoader configuration mapping. Mutually exclusive
            with ``reader``.
        reader : mapping, optional
            Reader configuration mapping. Mutually exclusive with
            ``loader``.
        writer : mapping, optional
            Writer configuration mapping. If provided with a loader, the
            loader output must be unwrapped before writing. This is enforced
            here because writer input shape depends on loader-batch unwrapping.
        rank : int, optional
            Process rank used by distributed loaders.
        dtype : str, default 'float32'
            Data type passed to loader datasets.
        world_size : int, default 0
            Number of devices/processes used by distributed loading.
        distributed : bool, default False
            If ``True``, initialize distributed data loading hooks.
        unwrap : bool, default False
            If ``True`` and using a loader, initialize an ``Unwrapper``. This
            is ignored for reader mode, which already loads one event at a
            time.
        iterations : int, optional
            Number of batches/entries to process. ``-1`` means one epoch.
        epochs : float, optional
            Number of epochs to process.
        split_output : bool, default False
            If ``True``, derive one output prefix per input file.

        Raises
        ------
        ValueError
            If neither or both of ``loader`` and ``reader`` are provided, or if
            both ``iterations`` and ``epochs`` are specified.
        """
        # Must provide exactly one of loader or reader configurations.
        if (loader is not None) == (reader is not None):
            raise ValueError(
                "Must provide either a loader or a reader configuration, not both."
            )

        # A bounded run can be expressed in iterations or epochs, but not both.
        # Both can be omitted for loading through Driver.__next__ or direct
        # Driver.process calls.
        if iterations is not None and epochs is not None:
            raise ValueError(
                "Must specify either `iterations` or `epochs` parameters, not both."
            )

        # Initialize attributes to default values before calling setup methods.
        self.watch = StopwatchManager()
        self.loader = None
        self.loader_iter = None
        self.unwrapper = None
        self.reader = None
        self.writer = None
        self.iterations = iterations
        self.epochs = epochs
        self.distributed = distributed

        # Initialize the loader or reader
        if loader is not None:
            self._initialize_loader(
                loader,
                rank=rank,
                dtype=dtype,
                world_size=world_size,
                distributed=distributed,
                unwrap=unwrap,
            )
        else:
            self._initialize_reader(reader)

        # Harmonize iteration/epoch configuration into explicit iterations.
        self._harmonize_iteration_count()

        # Derive log and output prefixes from the input file names.
        if self.reader is None:
            raise RuntimeError("I/O initialization did not produce a reader.")
        output_suffix = self._get_output_suffix(writer)
        self.log_prefix, self.output_prefix = self.get_prefixes(
            self.reader.file_paths,
            split_output,
            output_suffix=output_suffix,
        )

        # Initialize the writer, if configured.
        self._initialize_writer(writer, split_output=split_output, unwrap=unwrap)

    def _initialize_loader(
        self,
        loader: Mapping[str, Any],
        *,
        rank: int | None,
        dtype: str,
        world_size: int,
        distributed: bool,
        unwrap: bool,
    ) -> None:
        """Initialize the configured data loader.

        Parameters
        ----------
        loader : mapping
            Loader configuration mapping.
        rank : int, optional
            Process rank used by distributed samplers.
        dtype : str
            Floating-point precision requested by the driver.
        world_size : int
            Number of devices/processes used by the run.
        distributed : bool
            If ``True``, initialize loader components for distributed use.
        unwrap : bool
            If ``True``, initialize an ``Unwrapper`` and register the unwrap
            timer. Unwrapping is coupled to loader-backed I/O because it turns
            batches into the event-level data view expected downstream.

        Raises
        ------
        ImportError
            If PyTorch is unavailable.
        """
        # Loader functionality requires PyTorch, so check for it before proceeding.
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for loader functionality. "
                "Install with: pip install spine[model]"
            )

        # Initialize the loader and reader attributes and register the load timer.
        self.watch.initialize("load")
        self.loader = loader_factory(
            **loader,
            rank=rank,
            dtype=dtype,
            world_size=world_size,
            distributed=distributed,
        )
        self.iter_per_epoch = len(self.loader)
        self.reader = self.loader.dataset.reader

        # Initialize the unwrapper if requested. This stays with I/O because it
        # is only meaningful for loader-backed batches.
        if unwrap:
            self.watch.initialize("unwrap")
            self.unwrapper = Unwrapper()

        # If working from LArCV files, no post-processor was yet run.
        self.post_list = ()

    def _initialize_reader(self, reader: Mapping[str, Any] | None) -> None:
        """Initialize the configured reader.

        Parameters
        ----------
        reader : mapping
            Reader configuration mapping.
        """
        if reader is None:
            raise RuntimeError("Reader configuration is required in reader mode.")

        self.watch.initialize("read")
        self.reader = reader_factory(reader)
        self.iter_per_epoch = len(self.reader)

        # Post-processors already run on the input file, if available from the reader
        # metadata, are recorded here for use by the driver when determining which
        # post-processor information is already available.
        # TODO: this only works with two runs in a row, not 3 and above.
        self.post_list = None
        if self.reader.cfg is not None and "post" in self.reader.cfg:
            self.post_list = tuple(self.reader.cfg["post"])

    def _initialize_writer(
        self,
        writer: Mapping[str, Any] | None,
        *,
        split_output: bool,
        unwrap: bool,
    ) -> None:
        """Initialize the configured writer, if any.

        Parameters
        ----------
        writer : mapping, optional
            Writer configuration mapping.
        split_output : bool
            If ``True``, request one output file per input file.
        unwrap : bool
            If ``True``, loader batches are unwrapped before writing.

        Raises
        ------
        ValueError
            If writing loader output without unwrapping it first.
        """
        # No writer configured, skip initialization.
        if writer is None:
            return

        # Writers consume dictionaries with per-entry values; loader batches
        # must be unwrapped before they can be safely written.
        if self.loader is not None and not unwrap:
            raise ValueError("Must unwrap the model output to write it to file.")

        # Register the write timer and initialize the writer.
        self.watch.initialize("write")
        self.writer = writer_factory(
            writer, prefix=self.output_prefix, split=split_output
        )

    def _harmonize_iteration_count(self) -> None:
        """Convert iteration/epoch configuration into explicit iterations.

        ``iterations`` is the canonical runtime quantity used by the driver.
        Negative ``iterations`` means one complete pass over the input, while
        ``epochs`` is converted to an integer number of iterations using
        ``iter_per_epoch``.
        """
        if self.iterations is not None:
            if self.iterations < 0:
                self.iterations = self.iter_per_epoch
            self.epochs = 1.0
        elif self.epochs is not None:
            self.iterations = int(self.epochs * self.iter_per_epoch)

    def __len__(self) -> int:
        """Returns the number of events in the underlying reader object.

        Returns
        -------
        int
            Number of elements in the underlying loader/reader.
        """
        if self.reader is None:
            raise RuntimeError("Cannot determine length without an initialized reader.")
        return len(self.reader)

    @property
    def has_loader(self) -> bool:
        """Whether this manager is backed by a sequential loader."""
        return self.loader is not None

    @property
    def has_writer(self) -> bool:
        """Whether this manager owns an output writer."""
        return self.writer is not None

    def _name_max(self, path: str = ".") -> int:
        """Return the maximum filename component length for a path."""
        try:
            return os.pathconf(path, "PC_NAME_MAX")
        except (AttributeError, OSError, ValueError):
            return 255

    def _truncate_prefix(self, prefix: str, max_length: int) -> str:
        """Truncate a generated prefix while retaining both ends."""
        marker = "---"
        if len(prefix) <= max_length:
            return prefix

        if max_length <= len(marker):
            return marker[:max_length]

        keep = max_length - len(marker)
        head = (keep + 1) // 2
        tail = keep // 2
        return f"{prefix[:head]}{marker}{prefix[-tail:]}"

    def _get_output_suffix(self, writer: Mapping[str, Any] | None) -> str:
        """Return the suffix appended by a prefix-derived writer filename."""
        if writer is None or writer.get("file_name"):
            return ""

        writer_name = writer.get("name")
        default_suffixes = {"hdf5": "spine", "stage_hdf5": "stage"}
        if writer_name not in default_suffixes:
            return ""

        suffix = writer.get("suffix", default_suffixes[writer_name])
        return f"_{suffix}.h5"

    def format_log_name(self, log_name: str, log_dir: str) -> str:
        """Prefix and truncate a driver log name using the input-derived stem.

        Parameters
        ----------
        log_name : str
            Baseline log filename, such as ``spine_log.csv`` or
            ``train_log-0000001.csv``.
        log_dir : str
            Directory in which the log will be created. This is used to query
            the filesystem filename-component limit.

        Returns
        -------
        str
            Log filename prefixed by ``self.log_prefix`` and truncated, if
            needed, to fit the target filesystem constraints.
        """
        suffix = f"_{log_name}"
        max_length = max(1, self._name_max(log_dir) - len(suffix))
        log_prefix = self._truncate_prefix(self.log_prefix, max_length)
        return f"{log_prefix}{suffix}"

    def load(
        self,
        entry: int | None = None,
        run: int | None = None,
        subrun: int | None = None,
        event: int | None = None,
    ) -> dict[str, Any]:
        """Load one batch or reader entry.

        Parameters
        ----------
        entry : int, optional
            Entry number to load from a reader. This is not valid when using a
            loader.
        run : int, optional
            Run number to load from a reader.
        subrun : int, optional
            Subrun number to load from a reader.
        event : int, optional
            Event number to load from a reader.

        Returns
        -------
        dict[str, Any]
            Loaded batch or entry data. Loader-backed execution still returns
            one batched dictionary; per-entry lists are produced later by
            :meth:`unwrap`.

        Raises
        ------
        ValueError
            If direct entry addressing is requested from a loader, or if a
            reader request is incomplete.
        """
        self.watch.reset_if_active()

        if self.loader is not None:
            if (
                entry is not None
                or run is not None
                or subrun is not None
                or event is not None
            ):
                raise ValueError(
                    "When calling the loader, there is no way to request a "
                    "specific entry or run/subrun/event triplet, because the loader "
                    "is designed to load batches sequentially. Use the apply_filter "
                    "method to restrict the list of entries to load."
                )

            if self.loader_iter is None:
                self.loader_iter = iter(self.loader)

            self.watch.start("load")
            data = next(self.loader_iter)
            self.watch.stop("load")
            return data

        if entry is None and (run is None or subrun is None or event is None):
            raise ValueError(
                "Provide either the entry number or the run, subrun "
                "and event number to read."
            )

        if self.reader is None:
            raise RuntimeError("Cannot load data without an initialized reader.")

        self.watch.start("read")
        if entry is not None:
            data = self.reader.get(entry)
        else:
            data = self.reader.get_run_event(run, subrun, event)
        self.watch.stop("read")
        return data

    def reset_loader(self) -> None:
        """Reset the loader iterator, if this manager owns a loader."""
        if self.loader is not None:
            self.loader_iter = iter(self.loader)

    def prepare_iteration(self, iteration: int) -> None:
        """Prepare loader state for a driver iteration.

        Parameters
        ----------
        iteration : int
            Global driver iteration number. Loader-backed execution uses this
            to reset the iterator at epoch boundaries and to seed distributed
            samplers with their epoch number.
        """
        if self.loader is None:
            return

        if self.loader_iter is None or iteration % self.iter_per_epoch == 0:
            if self.distributed:
                epoch_cnt = iteration // self.iter_per_epoch
                self.loader.sampler.set_epoch(epoch_cnt)
            self.reset_loader()

    def unwrap(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert loader-batched values to per-entry values, if configured.

        Parameters
        ----------
        data : dict[str, Any]
            Loaded and optionally model-processed batch data.

        Returns
        -------
        dict[str, Any]
            Original data when no unwrapper is configured, otherwise unwrapped
            data products. The result remains a dictionary, but individual
            values may be converted from batched arrays/tensors into per-entry
            lists. This is a no-op for reader-backed I/O.
        """
        if self.unwrapper is None:
            return data

        self.watch.reset_if_active()
        self.watch.start("unwrap")
        result = self.unwrapper(data)
        self.watch.stop("unwrap")
        return result

    def write(
        self,
        data: dict[str, Any],
        cfg: Mapping[str, Any],
    ) -> None:
        """Write processed output, if a writer was configured.

        Parameters
        ----------
        data : dict[str, Any]
            Processed data products.
        cfg : Mapping[str, Any]
            Resolved driver configuration.
        """
        if self.writer is None:
            return

        self.watch.reset_if_active()
        self.watch.start("write")
        self.writer(data, cfg)
        self.watch.stop("write")

    def close(self) -> None:
        """Finalize and close the output writer, if any."""
        if self.writer is not None:
            self.writer.finalize()
            self.writer.close()

    def apply_filter(
        self,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: list[int] | None = None,
        skip_entry_list: list[int] | None = None,
        run_event_list: list[tuple[int, int, int]] | None = None,
        skip_run_event_list: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Restrict the reader entry list.

        Parameters
        ----------
        n_entry : int, optional
            Maximum number of entries to load.
        n_skip : int, optional
            Number of entries to skip at the beginning.
        entry_list : list[int], optional
            Entry IDs to keep.
        skip_entry_list : list[int], optional
            Entry IDs to skip.
        run_event_list : list[tuple[int, int, int]], optional
            Run/subrun/event triplets to keep.
        skip_run_event_list : list[tuple[int, int, int]], optional
            Run/subrun/event triplets to skip.
        """
        if self.reader is None:
            raise RuntimeError("Cannot filter entries without an initialized reader.")

        self.reader.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
        )
        self.loader_iter = None

    def get_prefixes(
        self,
        file_paths: list[str] | tuple[str, ...],
        split_output: bool,
        output_suffix: str = "",
    ) -> tuple[str, str | list[str]]:
        """Build log and output prefixes from the list of input files.

        The log prefix summarizes the input set without trying to infer a
        shared semantic name from character-level common prefixes. For one
        unique input stem, use that stem. For two distinct stems, join the first
        and last stem. For larger sets, include the total number of files
        between the first and last stem. When ``split_output`` is enabled, the
        output prefix remains one stem per input file.

        Parameters
        ----------
        file_paths : list[str] or tuple[str, ...]
            List of input file paths. Directory names are ignored.
        split_output : bool
            If ``True``, return one output prefix per input file.
        output_suffix : str, default ''
            Suffix appended to each output prefix when deriving writer file
            names. Used to cap prefixes against the filesystem component limit.

        Returns
        -------
        log_prefix : str
            Shared input summary string used to prefix logs.
        output_prefix : str or list[str]
            Shared output prefix, or one prefix per input file when
            ``split_output`` is enabled.
        """
        if not file_paths:
            raise ValueError("Must provide at least one input file path.")

        file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
        unique_file_names = tuple(dict.fromkeys(file_names))

        sep = "--"
        if len(unique_file_names) == 1:
            log_prefix = unique_file_names[0]
        elif len(file_names) == 2:
            log_prefix = sep.join((file_names[0], file_names[-1]))
        else:
            log_prefix = sep.join(
                (file_names[0], f"{len(file_names)}files", file_names[-1])
            )

        raw_log_prefix = log_prefix
        max_length = self._name_max()
        output_max_length = max(1, max_length - len(output_suffix))
        log_prefix = self._truncate_prefix(raw_log_prefix, max_length)
        output_prefix = self._truncate_prefix(raw_log_prefix, output_max_length)
        output_prefixes = [
            self._truncate_prefix(file_name, output_max_length)
            for file_name in file_names
        ]

        if not split_output:
            return log_prefix, output_prefix
        return log_prefix, output_prefixes
