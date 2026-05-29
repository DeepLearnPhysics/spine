"""SPINE driver class.

Takes care of everything in one centralized place:
- Data loading
- ML model and loss forward pass
- Batch unwrapping
- Representation building
- Post-processing
- Analysis script execution
- Writing output to file
"""

import inspect
import os
import random
import subprocess as sc
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from .ana import AnaManager
from .construct import BuildManager
from .geo import GeoManager
from .io import IOManager
from .math import seed as numba_seed
from .model import ModelManager
from .post import PostManager
from .utils.conditional import TORCH_AVAILABLE
from .utils.log import LogManager
from .utils.logger import configure_rank_logging, logger
from .utils.stopwatch import StopwatchManager
from .utils.torch import runtime
from .utils.torch.devices import set_visible_devices
from .version import __version__

__all__ = ["Driver"]


class Driver:
    """Central SPINE driver.

    Processes global configuration and runs the appropriate modules:
      1. Load data
      2. Run the model forward (including loss) and backward (if training)
      3. Unwrap batched data
      4. Build representations
      5. Run post-processing
      6. Run analysis scripts
      7. Write to file

    It takes a configuration dictionary of the form:

    .. code-block:: yaml

        base:
          <Base driver configuration>
        geo:
          <Geometry configuration>
        io:
          <Input/output configuration>
        model:
          <Model architecture>
        build:
          <Rules as to how to build reconstructed object representations>
        post:
          <Post-processors>
        ana:
          <Analysis scripts>
    """

    # Base-configuration keys forwarded into :meth:`initialize_base`.
    DRIVER_BASE_KEYS: frozenset[str]

    # Base-configuration keys consumed by launcher/runtime setup code.
    RUNTIME_BASE_KEYS = frozenset({"gpus", "torch_sharing_strategy", "verbosity"})

    def __init__(self, cfg: dict[str, Any], rank: int | None = None) -> None:
        """Build a driver from a full SPINE configuration.

        Parameters
        ----------
        cfg : dict[str, Any]
            Full SPINE configuration dictionary. This must contain an ``io``
            section and may contain ``base``, ``geo``, ``model``, ``build``,
            ``post``, and ``ana`` sections.
        rank : int, optional
            Rank of the current process in distributed execution. ``None``
            indicates a single-process run or a launcher-managed rank that has
            not yet been assigned at driver construction time.
        """
        # Process the full configuration dictionary and store it
        base, io, geo, model, build, post, ana = self.process_config(**cfg, rank=rank)
        driver_base = self.extract_driver_base_config(base)

        # Initialize the timers and the configuration dictionary
        self.watch = StopwatchManager()
        self.watch.initialize("iteration")

        # Initialize the base driver configuration parameters
        train = self.initialize_base(**driver_base, rank=rank)

        # Initialize the detector geometry singleton once and for all modules
        self.initialize_geo(geo)

        # Initialize the input/output
        self.initialize_io(io)

        # Initialize the ML model
        self.initialize_model(model, train)

        # Initialize the data representation builder
        self.initialize_builder(build)

        # Initialize the post-processors
        self.initialize_post(post)

        # Initialize the analysis scripts
        self.initialize_ana(ana)

        # Place-holder for the structured log manager, initialized in run()
        self.log_manager = None

        # Initialize the counter for non-loader iteration
        self._entry_counter = 0

    def process_config(
        self,
        io: dict[str, Any] | None = None,
        base: dict[str, Any] | None = None,
        geo: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        build: dict[str, Any] | None = None,
        post: dict[str, Any] | None = None,
        ana: dict[str, Any] | None = None,
        rank: int | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """Normalize the configuration and record the resolved state.

        Parameters
        ----------
        io : dict[str, Any] | None, optional
            I/O configuration dictionary. This section is mandatory.
        base : dict[str, Any] | None, optional
            Base driver configuration dictionary.
        geo : dict[str, Any] | None, optional
            Geometry configuration dictionary.
        model : dict[str, Any] | None, optional
            Model configuration dictionary.
        build : dict[str, Any] | None, optional
            Representation-building configuration dictionary.
        post : dict[str, Any] | None, optional
            Post-processor configuration dictionary.
        ana : dict[str, Any] | None, optional
            Analysis script configuration dictionary.
        rank : int, optional
            Rank of the current process.

        Returns
        -------
        tuple
            Tuple containing the normalized ``base``, ``io``, ``geo``,
            ``model``, ``build``, ``post``, and ``ana`` configuration
            dictionaries in that order.
        """
        # Copy user-provided configuration blocks before normalizing them. The
        # driver stores the resolved configuration, but should not mutate the
        # object handed to it by the caller.
        base = dict(base or {})
        if io is None:
            raise ValueError("The `io` block must be provided in the configuration.")

        io = deepcopy(io)
        geo = deepcopy(geo)
        model = deepcopy(model)
        build = deepcopy(build)
        post = deepcopy(post)
        ana = deepcopy(ana)

        # Set the verbosity of the logger
        verbosity = base.get("verbosity", "info")
        logger.setLevel(verbosity.upper())

        # Suppress low-priority distributed logs from non-main ranks early.
        configure_rank_logging(rank)

        # Set GPUs visible to CUDA (function handles torch availability)
        base["world_size"] = set_visible_devices(
            world_size=base.get("world_size", None), gpus=base.get("gpus", None)
        )

        # Normalize the seed configuration
        self.normalize_seed_config(base, io)

        # Rebuild global configuration dictionary
        self.cfg = {"base": base, "io": io}
        if geo is not None:
            self.cfg["geo"] = geo
        if model is not None:
            self.cfg["model"] = model
        if build is not None:
            self.cfg["build"] = build
        if post is not None:
            self.cfg["post"] = post
        if ana is not None:
            self.cfg["ana"] = ana

        # Log environment information
        logger.info("SPINE version: %s\n", __version__)

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        logger.info("$CUDA_VISIBLE_DEVICES=%s\n", visible_devices)

        system_info = sc.getstatusoutput("uname -a")[1]
        logger.info("Configuration processed at: %s\n", system_info)

        # Log configuration
        logger.info(yaml.dump(self.cfg, default_flow_style=None, sort_keys=False))

        # Return updated configuration
        return base, io, geo, model, build, post, ana

    def normalize_seed_config(self, base: dict[str, Any], io: dict[str, Any]) -> None:
        """Normalize driver and sampler seed configuration in place.

        Parameters
        ----------
        base : dict[str, Any]
            Resolved base configuration dictionary.
        io : dict[str, Any]
            Resolved I/O configuration dictionary.
        """
        # Generate a seed based on the current time if one is not provided.
        generated_seed = int(time.time())

        # Set the random sampler seed in the loader configuration if it is not set or invalid.
        loader_cfg = io.get("loader")
        if loader_cfg is not None and "sampler" in loader_cfg:
            sampler_cfg = loader_cfg["sampler"]
            if isinstance(sampler_cfg, str):
                sampler_cfg: dict[str, Any] = {"name": sampler_cfg}
                loader_cfg["sampler"] = sampler_cfg
            elif not isinstance(sampler_cfg, dict):
                raise TypeError(
                    "The loader sampler configuration must be a string or "
                    f"dictionary, got: {type(sampler_cfg).__name__}"
                )

            if "seed" not in sampler_cfg or sampler_cfg["seed"] < 0:
                sampler_cfg["seed"] = generated_seed

        # Set the global driver seed if it is not set or invalid. This is used to seed the
        # random number generators for Python, NumPy, Numba, and PyTorch, and is also forwarded
        # into the model manager for use in model initialization and training.
        if "seed" not in base or base["seed"] < 0:
            base["seed"] = generated_seed
        elif not isinstance(base["seed"], int):
            raise TypeError(f"The driver seed must be an integer, got: {base['seed']}")

    @classmethod
    def extract_driver_base_config(cls, base: Mapping[str, Any]) -> dict[str, Any]:
        """Extract and validate the base keys owned by :class:`Driver`.

        Parameters
        ----------
        base : Mapping[str, Any]
            Resolved base configuration dictionary.

        Returns
        -------
        dict[str, Any]
            Subset of the base configuration used to initialize
            :class:`Driver` state.

        Notes
        -----
        Keys consumed by launcher/runtime code are permitted in ``base`` but
        are intentionally not forwarded into :meth:`initialize_base`. Any
        other key is treated as a configuration error and rejected.
        """
        allowed_keys = cls.DRIVER_BASE_KEYS | cls.RUNTIME_BASE_KEYS
        invalid_keys = sorted(set(base) - allowed_keys)
        if invalid_keys:
            invalid = ", ".join(invalid_keys)
            raise KeyError(f"Unrecognized keys in `base` configuration: {invalid}")

        return {
            key: value for key, value in base.items() if key in cls.DRIVER_BASE_KEYS
        }

    def initialize_base(
        self,
        seed: int,
        world_size: int,
        dtype: str = "float32",
        log_dir: str = "logs",
        prefix_log: bool = False,
        overwrite_log: bool = False,
        csv_buffer_size: int = 1,
        parent_path: str | None = None,
        iterations: int | None = None,
        epochs: float | None = None,
        unwrap: bool = False,
        rank: int | None = None,
        log_step: int = 1,
        distributed: bool = False,
        split_output: bool = False,
        train: dict[str, Any] | None = None,
        tensorboard: bool | Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Initialize the driver state derived from the ``base`` block.

        Parameters
        ----------
        seed : int
            Random number generator seed.
        world_size : int
            Number of visible accelerator devices available to the run.
        dtype : str, default 'float32'
            Floating-point dtype used by the model and numerical I/O paths.
        log_dir : str, default 'logs'
            Directory where CSV logs should be written.
        prefix_log : bool, default False
            If ``True``, prefix log file names with an input-derived stem.
        overwrite_log : bool, default False
            If ``True``, allow the CSV writer to overwrite an existing log.
        csv_buffer_size : int, default 1
            CSV file buffer size. 1 is line buffered (default, safe),
            -1 uses system default, 0 is unbuffered, >1 is buffer size in bytes
        parent_path : str, optional
            Parent path used to resolve relative analysis-script paths.
        iterations : int, optional
            Number of entries or batches to process. ``None`` means use the
            full dataset/loader.
        epochs : float, optional
            Number of passes over the full dataset when iterating with a
            loader.
        unwrap : bool, default False
            If ``True``, unwrap batched data into per-entry outputs.
        rank : int, optional
            Rank of the current process in distributed execution.
        log_step : int, default 1
            Logging period in iterations.
        distributed : bool, default False
            If ``True``, mark this process as participating in distributed
            execution.
        split_output : bool, default False
            If ``True``, write one output file per input file.
        train : dict[str, Any] | None, optional
            Training configuration dictionary. This method does not interpret
            the content; it returns it so the model manager can do so.
        tensorboard : bool | Mapping[str, Any] | None, optional
            TensorBoard logging configuration. ``False`` or ``None`` disable
            TensorBoard logging, ``True`` uses default settings, and a mapping
            overrides defaults such as output directory and flush interval.

        Returns
        -------
        dict[str, Any] | None
            Training configuration dictionary to forward into the model
            manager, if any.
        """
        # Set up the seed
        random.seed(seed)
        np.random.seed(seed)
        numba_seed(seed)
        runtime.manual_seed(seed)

        # Set up the device the model will run on
        if rank is None and world_size > 0:
            if world_size >= 2:
                raise ValueError(
                    "Must not request > 1 GPU without specifying a GPU rank."
                )
            rank = 0

        self.rank = rank
        self.world_size = world_size
        self.main_process = rank is None or rank == 0

        # Check on the distributed process
        if self.rank is not None and self.rank >= world_size:
            raise ValueError(
                f"The GPU rank index of this driver ({rank}) is too large "
                f"for the number of GPUs available ({world_size})."
            )

        self.distributed = distributed
        if not distributed and world_size > 1:
            self.distributed = True

        # Store general parameters
        self.dtype = dtype
        self.log_dir = log_dir
        self.prefix_log = prefix_log
        self.overwrite_log = overwrite_log
        self.csv_buffer_size = csv_buffer_size
        self.parent_path = parent_path
        self.iterations = iterations
        self.epochs = epochs
        self.unwrap = unwrap
        self.seed = seed
        self.log_step = log_step
        self.split_output = split_output
        self.tensorboard_cfg = tensorboard

        return train

    def initialize_io(self, io: Mapping[str, Any]) -> None:
        """Initialize the input/output manager.

        Parameters
        ----------
        io : Mapping[str, Any]
            Top-level I/O configuration mapping. This may contain ``loader``,
            ``reader``, and/or ``writer`` sections.
        """
        self.io = IOManager(
            **io,
            rank=self.rank,
            dtype=self.dtype,
            world_size=self.world_size,
            distributed=self.distributed,
            unwrap=self.unwrap,
            iterations=self.iterations,
            epochs=self.epochs,
            split_output=self.split_output,
        )

        # Keep only high-level scheduling state on the driver; I/O resources
        # remain owned by IOManager.
        self.iterations = self.io.iterations
        self.epochs = self.io.epochs

    def initialize_geo(self, geo: Mapping[str, Any] | None = None) -> None:
        """Initialize the detector geometry singleton.

        Parameters
        ----------
        geo : Mapping[str, Any] | None, optional
            Geometry configuration mapping. If ``None``, geometry-dependent
            modules are left uninitialized until they are explicitly requested.
        """
        if geo is not None:
            GeoManager.initialize_or_get(**geo)

    def initialize_model(
        self,
        model: Mapping[str, Any] | None = None,
        train: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the model manager, if requested.

        Parameters
        ----------
        model : Mapping[str, Any] | None, optional
            Model configuration mapping.
        train : Mapping[str, Any] | None, optional
            Training configuration mapping extracted from the base block.

        Notes
        -----
        A model requires a loader-backed input pipeline. If a ``train`` block
        is provided without a model block, initialization fails because there
        is no model to optimize.
        """
        self.model = None
        if model is None:
            if train is not None:
                raise ValueError(
                    "Received a train block but there is no model to train."
                )
            return

        if not self.io.has_loader:
            raise ValueError("The model can only be used in conjunction with a loader.")

        self.watch.initialize("model")

        # Check if PyTorch is available for model functionality
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for model functionality. "
                "Install with: pip install spine[model]"
            )

        self.model = ModelManager(
            **model,
            train=dict(train) if train is not None else None,
            dtype=self.dtype,
            rank=self.rank,
            distributed=self.distributed,
            iter_per_epoch=self.io.iter_per_epoch,
        )

    def initialize_builder(self, build: Mapping[str, Any] | None = None) -> None:
        """Initialize reconstructed/truth representation building.

        Parameters
        ----------
        build : Mapping[str, Any] | None, optional
            Representation-building configuration mapping.

        Notes
        -----
        Builder execution happens after optional model forwarding and optional
        unwrapping. If a model is present, its output must be unwrapped and
        converted to NumPy before representations can be built.
        """
        self.builder = None
        if build is None:
            return

        if self.model is not None and not self.unwrap:
            raise ValueError("Must unwrap the model output to build representations.")

        if self.model is not None and not self.model.to_numpy:
            raise ValueError(
                "Must cast model output to numpy to build representations."
            )

        self.watch.initialize("build")
        self.builder = BuildManager(**build)

    def initialize_post(self, post: Mapping[str, Any] | None = None) -> None:
        """Initialize post-processing modules.

        Parameters
        ----------
        post : Mapping[str, Any] | None, optional
            Post-processing configuration mapping.

        Notes
        -----
        Post-processors operate on per-entry data products. When used after a
        model, the model output must therefore be unwrapped first.
        """
        self.post = None
        if post is None:
            return

        if self.model is not None and not self.unwrap:
            raise ValueError("Must unwrap the model output to run post-processors.")

        self.watch.initialize("post")
        self.post = PostManager(
            dict(post),
            post_list=self.io.post_list,
            parent_path=self.parent_path,
        )

    def initialize_ana(self, ana: Mapping[str, Any] | None = None) -> None:
        """Initialize analysis scripts.

        Parameters
        ----------
        ana : Mapping[str, Any] | None, optional
            Analysis configuration mapping.

        Notes
        -----
        Analysis scripts run on the same per-entry view of the data as
        post-processors. When used after a model, the model output must be
        unwrapped first.
        """
        self.ana = None
        if ana is None:
            return

        if self.model is not None and not self.unwrap:
            raise ValueError("Must unwrap the model output to run analysis scripts.")

        self.watch.initialize("ana")
        self.ana = AnaManager(
            dict(ana), log_dir=self.log_dir, prefix=self.io.log_prefix
        )

    def initialize_log(self) -> None:
        """Initialize CSV and optional TensorBoard logging backends."""
        # Make a directory if it does not exist
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Determine the log name, initialize it
        if self.builder is not None or self.model is None:
            # If running the driver more than a model, give a generic name
            log_name = "spine_log.csv"
        else:
            # If running the driver within a training/validation process
            # (model only), follow a specific pattern of log names.
            start_iteration = self.model.start_iteration
            prefix = "train" if self.model.train else "inference"
            suffix = "" if not self.model.distributed else f"_proc{self.rank}"
            log_name = f"{prefix}{suffix}_log-{start_iteration:07d}.csv"

        # If requested, prefix the log name with the input file name
        if self.prefix_log:
            log_name = self.io.format_log_name(log_name, self.log_dir)

        # Initialize the log
        log_path = os.path.join(self.log_dir, log_name)
        tb_dir = os.path.join(self.log_dir, "tensorboard")
        self.log_manager = LogManager(
            log_path,
            overwrite=self.overwrite_log,
            buffer_size=self.csv_buffer_size,
            tensorboard=self.tensorboard_cfg,
            tensorboard_dir=tb_dir,
        )

    def __len__(self) -> int:
        """Returns the number of events in the underlying reader object.

        Returns
        -------
        int
            Number of elements in the underlying loader/reader.
        """
        return len(self.io)

    def __iter__(self) -> "Driver":
        """Resets the counter and returns itself.

        Returns
        -------
        object
            The Driver itself
        """
        # If a loader is used, reinitialize it. Otherwise set an entry counter
        self._entry_counter = 0
        self.io.reset_loader()

        return self

    def __next__(self) -> dict[str, Any]:
        """Defines how to process the next entry in the iterator.

        Returns
        -------
        dict[str, Any]
            Processed data dictionary. If loader output was unwrapped, values
            inside the dictionary may be per-entry lists.
        """
        # If there are more iterations to go through, return data
        if self.io.has_loader:
            return self.process()
        else:
            if self._entry_counter < len(self):
                data = self.process(self._entry_counter)
                self._entry_counter += 1

                return data

            raise StopIteration

    def run(self) -> None:
        """Loop over the requested number of iterations, process them."""
        # To run the loop, must know how many times it must be done
        if self.iterations is None:
            raise ValueError("Must specify either `iterations` or `epochs` parameters.")

        # Initialize the output log
        self.initialize_log()

        try:
            # Get the iteration start (if model exists)
            start_iteration = 0
            if self.model is not None and self.model.train:
                start_iteration = self.model.start_iteration

            # Loop and process each iteration
            for iteration in range(start_iteration, self.iterations):
                # Let I/O prepare loader state, if using a loader.
                self.io.prepare_iteration(iteration)

                # Update the epoch counter, record the execution date/time
                epoch = (iteration + 1) / self.io.iter_per_epoch
                tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Process one batch/entry of data
                entry = None if self.io.has_loader else iteration
                data = self.process(entry=entry, iteration=iteration, epoch=epoch)

                # Log the output
                self.log(data, tstamp, iteration, epoch)

                # Release the memory for the next iteration
                data = None
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Close output resources owned by the driver."""
        if self.ana is not None:
            self.ana.close()
        if self.log_manager is not None:
            self.log_manager.close()
        if hasattr(self, "io"):
            self.io.close()

    def process(
        self,
        entry: int | None = None,
        run: int | None = None,
        subrun: int | None = None,
        event: int | None = None,
        iteration: int | None = None,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Process one entry or a batch of entries.

        Run single step of main SPINE driver. This includes data loading,
        model forwarding, data structure building, post-processing
        and appending desired information to each row of output csv files.

        Parameters
        ----------
        entry : int, optional
            Entry number to load
        run : int, optional
            Run number to load
        subrun : int, optional
            Subrun number to load
        event : int, optional
            Event number to load
        iteration : int, optional
            Iteration number. Only needed to train models and/or to apply
            time-dependant model losses, no-op otherwise
        epoch : float, optional
            Epoch fraction. Only needed to train models, no-op otherwise

        Returns
        -------
        dict[str, Any]
            Processed data dictionary. If loader output was unwrapped, values
            inside the dictionary may be per-entry lists.
        """
        # 0. Make sure there is no watch running, start the iteration timer
        for watch in self.watch.values():
            if watch.running or watch.paused:
                self.watch.reset()
                break

        self.watch.start("iteration")

        # 1. Load data
        data = self.io.load(entry, run, subrun, event)

        # 2. Pass data through the model
        if self.model is not None:
            self.watch.start("model")
            result = self.model(data, iteration=iteration, epoch=epoch)
            data.update(**result)
            self.watch.stop("model")
            self.watch.update(self.model.watch, "model")

        # 3. Unwrap
        data = self.io.unwrap(data)

        # 4. Build representations
        if self.builder is not None:
            self.watch.start("build")
            self.builder(data)
            self.watch.stop("build")

        # 5. Run post-processing, if requested
        if self.post is not None:
            self.watch.start("post")
            self.post(data)
            self.watch.stop("post")
            self.watch.update(self.post.watch, "post")

        # 6. Run scripts, if requested
        if self.ana is not None:
            self.watch.start("ana")
            self.ana(data)
            self.watch.stop("ana")
            self.watch.update(self.ana.watch, "ana")

        # 7. Write output to file, if requested
        self.io.write(data, self.cfg)
        self.watch.update(self.io.watch)

        # Stop the iteration timer
        self.watch.stop("iteration")

        # Return
        return data

    def apply_filter(
        self,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: list[int] | None = None,
        skip_entry_list: list[int] | None = None,
        run_event_list: list[tuple[int, int, int]] | None = None,
        skip_run_event_list: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Restrict the list of entries.

        Parameters
        ----------
        n_entry : int, optional
            Maximum number of entries to load
        n_skip : int, optional
            Number of entries to skip at the beginning
        entry_list : list, optional
            List of integer entry IDs to add to the index
        skip_entry_list : list, optional
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to skip from the index
        """
        self.io.apply_filter(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
        )

    def log(
        self,
        data: dict[str, Any],
        tstamp: str,
        iteration: int,
        epoch: float | None = None,
    ) -> None:
        """Log relevant information to CSV files and stdout.

        Parameters
        ----------
        data : dict
            Dictionary of data products to extract scalars from
        tstamp : str
            Time when this iteration was run
        iteration : int
            Iteration counter
        epoch : float
            Progress in the training process in number of epochs
        """
        # Check that the log manager is not being used before initialization
        if self.log_manager is None:
            raise RuntimeError("The log manager must be initialized before logging.")

        log_row = self.log_manager.append(data, self.watch, iteration, epoch)

        if self.should_log_stdout(iteration):
            self.log_manager.log_stdout_summary(
                log_row,
                data,
                self.watch,
                tstamp,
                iteration,
                epoch,
                model_train=self.model is not None and self.model.train,
                rank=self.rank,
                distributed=self.distributed,
                main_process=self.main_process,
            )

    def should_log_stdout(self, iteration: int) -> bool:
        """Return ``True`` when a formatted stdout summary should be emitted."""
        return ((iteration + 1) % self.log_step) == 0


Driver.DRIVER_BASE_KEYS = frozenset(
    name
    for name, parameter in inspect.signature(Driver.initialize_base).parameters.items()
    if name not in {"self", "rank"}
    and parameter.kind
    in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
)
