"""Input/output manager used by the central SPINE driver."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.unwrap import Unwrapper

from .factories import loader_factory, reader_factory, writer_factory

__all__ = ["IOManager"]


class IOManager:
    """Central I/O setup manager.

    The manager owns the I/O-specific setup rules used by
    :class:`spine.driver.Driver`. It selects exactly one input path, either a
    PyTorch-style loader or an entry reader, exposes the underlying reader used
    for event addressing, derives log/output prefixes from input file names,
    configures the optional writer and harmonizes ``iterations`` and ``epochs``.

    Attributes
    ----------
    loader : object or None
        Instantiated PyTorch DataLoader-like object. ``None`` when running from
        a reader.
    loader_iter : iterator or None
        Loader iterator used by the driver for sequential batch access.
    counter : int or None
        Entry counter used by the driver for reader-based iteration.
    unwrapper : Unwrapper or None
        Batch unwrapper initialized when requested for loader-based execution.
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
        watch: Any,
        geo: Mapping[str, Any] | None = None,
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
            loader output must be unwrapped before writing.
        watch : StopwatchManager
            Driver stopwatch manager used to register I/O timers.
        geo : mapping, optional
            Geometry configuration passed through to loader datasets.
        rank : int, optional
            Process rank used by distributed loaders.
        dtype : str, default 'float32'
            Data type passed to loader datasets.
        world_size : int, default 0
            Number of devices/processes used by distributed loading.
        distributed : bool, default False
            If ``True``, initialize distributed data loading hooks.
        unwrap : bool, default False
            If ``True`` and using a loader, initialize an ``Unwrapper``.
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
            raise ValueError("Must provide either a loader or a reader configuration.")

        # Must provide at most one of iterations or epochs.
        if iterations is not None and epochs is not None:
            raise ValueError(
                "Must not specify both `iterations` or `epochs` parameters."
            )

        # Initialize attributes to default values before calling setup methods.
        self.watch = watch
        self.loader = None
        self.loader_iter = None
        self.counter = None
        self.unwrapper = None
        self.reader = None
        self.writer = None
        self.iterations = iterations
        self.epochs = epochs

        # Initialize the loader or reader
        if loader is not None:
            self._initialize_loader(
                loader,
                geo=geo,
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
        geo: Mapping[str, Any] | None,
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
        geo : mapping, optional
            Geometry configuration passed through to the dataset factory.
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
            timer.

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
            geo=geo,
            rank=rank,
            dtype=dtype,
            world_size=world_size,
            distributed=distributed,
        )
        self.iter_per_epoch = len(self.loader)
        self.reader = self.loader.dataset.reader

        # Initialize the unwrapper if requested, and register the unwrap timer.
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
        assert reader is not None  # For the type checker, guaranteed by the caller.
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

        # If a writer is configured with a loader, the loader output must be unwrapped.
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
