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
import platform
import random
import time
import warnings
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from numbers import Real
from typing import Any

import numpy as np
import yaml

from .ana import AnaManager
from .banner import BANNER_SEPARATOR
from .config import normalize_config
from .construct import BuildManager
from .geo import GeoManager
from .io import IOManager
from .logging import LogManager, configure_rank_logging, logger
from .math import seed as numba_seed
from .model import ModelManager, ValidationManager
from .post import PostManager
from .utils.conditional import TORCH_AVAILABLE, torch
from .utils.stopwatch import StopwatchManager
from .utils.torch import runtime
from .utils.torch.devices import set_visible_devices

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
        train:
          <Training regimen and checkpoint schedule>
        validation:
          <Checkpoint-bound validation and early stopping>
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
        # Normalize legacy block locations before dispatching top-level config.
        cfg = normalize_config(cfg)

        # Process the full configuration dictionary and store it
        (
            base,
            io,
            geo,
            model,
            train,
            validation,
            build,
            post,
            ana,
        ) = self.process_config(**cfg, rank=rank)
        driver_base = self.extract_driver_base_config(base)

        # Initialize the timers and the configuration dictionary
        self.watch = StopwatchManager()
        self.watch.initialize("iteration")

        # Initialize the base driver configuration parameters
        self.initialize_base(**driver_base, rank=rank)
        if train is not None and self.distributed and not self.ddp:
            raise ValueError("Distributed training requires `ddp: true`.")

        # Report the resolved run context before initializing heavier modules.
        self._log_startup(train is not None)

        # Initialize the detector geometry singleton once and for all modules
        self.initialize_geo(geo)

        # Initialize the input/output
        self.initialize_io(io)

        # Initialize the ML model
        self.initialize_model(model, train)

        # Initialize checkpoint-bound validation against the training model
        self.initialize_validation(validation, io)

        # Initialize the data representation builder
        self.initialize_builder(build)

        # Initialize the post-processors
        self.initialize_post(post)

        # Initialize the analysis scripts
        self.initialize_ana(ana)

        # Restore stochastic and loader state after every configured module is
        # constructed, so initialization cannot advance the resumed streams.
        self.restore_training_runtime()

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
        train: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
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
        train : dict[str, Any] | None, optional
            Top-level training regimen configuration.
        validation : dict[str, Any] | None, optional
            Checkpoint-bound validation configuration.
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
            ``model``, ``train``, ``validation``, ``build``, ``post``, and
            ``ana`` configuration dictionaries in that order.
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
        train = deepcopy(train)
        validation = deepcopy(validation)
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
        if train is not None:
            self.cfg["train"] = train
        if validation is not None:
            self.cfg["validation"] = validation
        if build is not None:
            self.cfg["build"] = build
        if post is not None:
            self.cfg["post"] = post
        if ana is not None:
            self.cfg["ana"] = ana

        # Return updated configuration
        return base, io, geo, model, train, validation, build, post, ana

    def _log_startup(self, training: bool) -> None:
        """Log the resolved runtime context and complete configuration.

        The command-line identity banner is intentionally absent here so that
        programmatic construction of :class:`Driver` remains visually compact.
        Rank-aware logging suppresses this report on non-primary workers.

        Parameters
        ----------
        training : bool
            Whether the resolved configuration contains a training regimen.
        """
        # Describe the effective compute target rather than merely echoing the
        # raw CUDA visibility environment variable.
        if self.world_size > 0:
            device_index = 0 if self.rank is None else self.rank
            device = f"cuda:{device_index}"
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices is not None:
                device += f" (CUDA_VISIBLE_DEVICES={visible_devices})"
        else:
            device = "cpu"

        runtime_lines = [
            f"Mode:          {'training' if training else 'inference'}",
            f"Host:          {platform.node() or 'unknown'}",
            f"Python:        {platform.python_version()}",
            f"Device:        {device}",
            f"World size:    {max(1, self.world_size)}",
            f"Seed:          {self.seed}",
        ]
        if self.distributed:
            runtime_lines.insert(5, f"Rank:          {self.rank}")

        config = yaml.dump(self.cfg, default_flow_style=False, sort_keys=False).rstrip()
        logger.info(
            "Runtime\n-------\n%s\n\nResolved configuration\n"
            "----------------------\n%s\n\n%s\n",
            "\n".join(runtime_lines),
            config,
            BANNER_SEPARATOR,
        )

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
        ddp: bool | None = None,
        split_output: bool = False,
        tensorboard: bool | Mapping[str, Any] | None = None,
    ) -> None:
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
        ddp : bool, optional
            Whether to wrap the model in DistributedDataParallel. Defaults to
            the distributed execution setting. Distributed training requires
            this to be enabled; distributed inference may disable it while
            retaining rank-based data sharding.
        split_output : bool, default False
            If ``True``, write one output file per input file.
        tensorboard : bool | Mapping[str, Any] | None, optional
            TensorBoard logging configuration. ``False`` or ``None`` disable
            TensorBoard logging, ``True`` uses default settings, and a mapping
            overrides defaults such as output directory and flush interval.

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
        self.ddp = self.distributed if ddp is None else ddp
        if self.ddp and not self.distributed:
            raise ValueError("`ddp` requires distributed execution.")
        # Store general parameters
        self.dtype = dtype
        self.log_dir = log_dir
        self.prefix_log = prefix_log
        self.overwrite_log = overwrite_log
        self.csv_buffer_size = csv_buffer_size
        self.parent_path = parent_path
        self.iterations = iterations
        self.epochs = epochs
        self.epoch_based = epochs is not None
        self.unwrap = unwrap
        self.seed = seed
        self.log_step = log_step
        self.split_output = split_output
        self.tensorboard_cfg = tensorboard

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
            Top-level training configuration mapping.

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
                "Use the released SPINE container or install a compatible "
                "PyTorch ecosystem manually."
            )

        self.model = ModelManager(
            **model,
            train=dict(train) if train is not None else None,
            dtype=self.dtype,
            rank=self.rank,
            distributed=getattr(self, "ddp", getattr(self, "distributed", False)),
            iter_per_epoch=self.io.iter_per_epoch,
        )

    def restore_training_runtime(self) -> None:
        """Restore this rank's RNG and loader state from a resume checkpoint.

        A runtime snapshot is meaningful only for the same distributed world
        size that produced it. Model, optimizer and scheduler state remain
        usable when the world changes, but rank-local stochastic streams and
        data shards cannot be mapped exactly and are intentionally skipped.
        """
        if self.model is None or not getattr(self.model, "resume_training", False):
            return
        state = getattr(self.model, "checkpoint_runtime_state", None)
        if state is None:
            return

        world_size = max(1, self.world_size)
        checkpoint_world_size = int(state.get("world_size", 1))
        if checkpoint_world_size != world_size:
            warnings.warn(
                "Checkpoint runtime state was recorded with world size "
                f"{checkpoint_world_size}, but this run uses {world_size}; RNG "
                "and loader state will not be restored exactly.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        rank = 0 if self.rank is None else self.rank
        rank_states = state.get("ranks", [])
        local_state = next(
            (item for item in rank_states if int(item.get("rank", -1)) == rank),
            None,
        )
        if local_state is None:
            warnings.warn(
                f"Checkpoint has no runtime state for rank {rank}; continuation "
                "will not be bit-for-bit exact.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        io_state = local_state.get("io")
        if io_state is not None:
            next_iteration = int(io_state.get("next_iteration", -1))
            if next_iteration != self.model.start_iteration:
                raise ValueError(
                    "Checkpoint loader cursor does not match its global step: "
                    f"expected iteration {self.model.start_iteration}, got "
                    f"{next_iteration}."
                )

        runtime.restore_rng_state(local_state["rng"])
        if io_state is not None:
            self.io.restore_checkpoint_state(io_state)

    def initialize_validation(
        self,
        validation: Mapping[str, Any] | None,
        io: Mapping[str, Any],
    ) -> None:
        """Initialize checkpoint-bound validation, if configured.

        Validation inherits the training loader schema and runtime settings,
        replacing only its dataset sources and stochastic sampling behavior.
        It reuses the live model so distributed training and validation share
        the same ranks, devices and DDP process group.

        Parameters
        ----------
        validation : mapping, optional
            Validation source, fraction and early-stopping configuration.
        io : mapping
            Training I/O configuration from which to derive the loader.

        Raises
        ------
        ValueError
            If validation is requested outside training, without a checkpoint
            cadence, or without a loader-backed input pipeline.
        """
        self.validation = None
        if validation is None:
            if (
                self.model is not None
                and getattr(self.model, "lr_scheduler_monitor", None) is not None
            ):
                raise ValueError(
                    "A monitored checkpoint scheduler requires a `validation` block."
                )
            return
        if self.model is None or not self.model.train:
            raise ValueError("On-the-fly validation requires a training model.")
        if self.model.save_step is None:
            raise ValueError(
                "On-the-fly validation requires `train.save_step` or "
                "`train.save_epoch`."
            )

        loader = io.get("loader")
        if not isinstance(loader, Mapping):
            raise ValueError("On-the-fly validation requires `io.loader`.")

        self.validation = ValidationManager(
            validation,
            loader,
            self.model,
            rank=self.rank,
            dtype=self.dtype,
            world_size=self.world_size,
            distributed=self.distributed,
            seed=self.seed,
            log_dir=self.log_dir,
            prefix_log=self.prefix_log,
            overwrite_log=self.overwrite_log,
            csv_buffer_size=self.csv_buffer_size,
            log_step=self.log_step,
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
            self.io.set_post_processors(())
            return

        if self.model is not None and not self.unwrap:
            raise ValueError("Must unwrap the model output to run post-processors.")

        self.watch.initialize("post")
        self.post = PostManager(
            dict(post),
            post_list=self.io.post_list,
            parent_path=self.parent_path,
        )
        self.io.set_post_processors(self.post.module_names)

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
        # If analysis scripts are not requested, skip initialization. Columnar
        # reader mode, however, requires at least one analysis script.
        self.ana = None
        if ana is None:
            if getattr(self.io, "columnar", False):
                raise ValueError(
                    "Columnar reader mode requires at least one analysis script."
                )
            return

        # Check for incompatible modules when columnar reader mode is requested
        if getattr(self.io, "columnar", False):
            incompatible = []
            if self.model is not None:
                incompatible.append("model")
            if self.builder is not None:
                incompatible.append("build")
            if self.post is not None:
                incompatible.append("post")
            if self.io.has_writer:
                incompatible.append("writer")
            if incompatible:
                raise ValueError(
                    "Columnar reader mode currently supports analysis-only "
                    f"workflows; remove: {', '.join(incompatible)}."
                )

        # Check the information is unwrapped before running analysis scripts
        if self.model is not None and not self.unwrap:
            raise ValueError("Must unwrap the model output to run analysis scripts.")

        # Initialize the analysis manager
        self.watch.initialize("ana")
        columnar = getattr(self.io, "columnar", False)
        self.ana = AnaManager(
            dict(ana),
            log_dir=self.log_dir,
            prefix=self.io.log_prefix,
            columnar=columnar,
        )

        # If columnar reader mode is requested, configure the I/O manager to request
        # the necessary columns needed by the analysis scripts.
        if columnar:
            self.io.configure_columnar(self.ana.columnar_requests())

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
            distributed = getattr(
                self, "distributed", getattr(self.model, "distributed", False)
            )
            suffix = "" if not distributed else f"_proc{self.rank}"
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
            tensorboard=(
                self.tensorboard_cfg if getattr(self, "main_process", True) else None
            ),
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

        success = False
        try:
            # Get the iteration start (if model exists)
            start_iteration = 0
            start_epoch = 0.0
            if self.model is not None and self.model.train:
                start_iteration = self.model.start_iteration
                start_epoch = getattr(self.model, "start_epoch", None)

            # Anchor loader epochs to checkpoint progress rather than the old
            # global iteration-to-batch-size relationship.
            if start_epoch is not None and hasattr(self.io, "set_resume_progress"):
                self.io.set_resume_progress(start_iteration, start_epoch)

            # Epoch limits describe total training progress. Re-express the
            # remaining epochs using the current loader's batch size.
            stop_iteration = self.iterations
            epochs = getattr(self, "epochs", None)
            if epochs is not None and start_epoch is not None:
                remaining_epochs = max(0.0, epochs - start_epoch)
                stop_iteration = start_iteration + int(
                    remaining_epochs * self.io.iter_per_epoch
                )

            # Loop and process each iteration
            for iteration in range(start_iteration, stop_iteration):
                # Let I/O prepare loader state, if using a loader.
                self.io.prepare_iteration(iteration)

                # Update the epoch counter, record the execution date/time
                if start_epoch is None:
                    epoch = (iteration + 1) / self.io.iter_per_epoch
                else:
                    relative_iteration = iteration - start_iteration + 1
                    epoch = start_epoch + relative_iteration / self.io.iter_per_epoch
                tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Process one batch/entry of data
                entry = None if self.io.has_loader else iteration
                data = self.process(entry=entry, iteration=iteration, epoch=epoch)

                # Report globally representative training scalars under DDP.
                self.reduce_training_metrics(data)

                # Checkpoint boundaries are driver-owned so all ranks can
                # validate before rank zero serializes the live weights.
                stop_training = False
                should_checkpoint = (
                    self.model is not None
                    and self.model.train
                    and self.model.should_save(iteration)
                )
                if should_checkpoint:
                    # Present the completed training step before entering the
                    # checkpoint section. The authoritative CSV row is still
                    # appended afterward so it includes checkpoint timings.
                    self.log_stdout(data, tstamp, iteration, epoch)

                    if self.main_process:
                        validation_batches = (
                            None
                            if self.validation is None
                            else self.validation.num_iterations
                        )
                        LogManager.log_checkpoint_start(
                            iteration,
                            epoch,
                            validation_batches,
                            self.distributed,
                        )

                    validation_state = None
                    validation_metrics = None
                    promote_best = False
                    if self.validation is not None:
                        validation_metrics = self.validation.run(iteration, epoch)
                        self.log_manager.append_tensorboard(
                            {
                                f"val_{key}": value
                                for key, value in validation_metrics.items()
                            },
                            iteration,
                        )
                        promote_best = self.validation.update_best_checkpoint(
                            validation_metrics
                        )
                        stop_training = self.validation.update_early_stopping(
                            validation_metrics
                        )
                        validation_state = self.validation.checkpoint_state(
                            validation_metrics
                        )

                    # Checkpoint-bound schedulers advance before their state is saved.
                    self.model.step_checkpoint_scheduler(validation_metrics)

                    # Every rank contributes its stochastic and loader cursor
                    # state before rank zero writes the shared checkpoint.
                    local_runtime_state = {
                        "rank": 0 if self.rank is None else self.rank,
                        "rng": runtime.capture_rng_state(),
                        "io": self.io.checkpoint_state(iteration + 1),
                    }
                    rank_states = runtime.distributed_all_gather_object(
                        local_runtime_state
                    )
                    checkpoint_runtime = {
                        "world_size": max(1, self.world_size),
                        "ranks": rank_states,
                    }

                    if self.main_process:
                        # Retain model-owned save timing around serialization
                        LogManager.log_checkpoint_saving()
                        self.model.watch.start("save")
                        datasets = {"train": self.io.dataset_provenance()}
                        if self.validation is not None:
                            datasets["validation"] = (
                                self.validation.io.dataset_provenance()
                            )
                        checkpoint_path = self.model.save_state(
                            iteration,
                            epoch,
                            validation_state,
                            config=self.cfg,
                            datasets=datasets,
                            runtime_state=checkpoint_runtime,
                            world_size=max(1, self.world_size),
                        )
                        if promote_best:
                            assert self.validation is not None
                            assert self.validation.best_checkpoint is not None
                            best_path = self.validation.best_checkpoint.path
                            self.model.save_best_state(checkpoint_path, best_path)
                        self.model.watch.stop("save")
                        self.watch.update(self.model.watch, "model")
                        LogManager.log_checkpoint_complete(
                            checkpoint_path,
                            self.distributed,
                            best_path if promote_best else None,
                        )

                # Log the output
                self.log(
                    data,
                    tstamp,
                    iteration,
                    epoch,
                    stdout=not should_checkpoint,
                )

                # Release the memory for the next iteration
                data = None
                if stop_training:
                    logger.info(
                        "Early stopping triggered at iteration %d (epoch %.4f).",
                        iteration,
                        epoch,
                    )
                    break
            success = True
        finally:
            self.cleanup(finalize_writer=success)

    def cleanup(self, finalize_writer: bool = True) -> None:
        """Close resources and finalize writer output after successful work.

        Parameters
        ----------
        finalize_writer : bool, default True
            If `True`, mark writer output complete before closing it. Exception
            paths pass `False` so partial output remains explicitly incomplete.
        """
        if self.ana is not None:
            self.ana.close()
        if self.log_manager is not None:
            self.log_manager.close()
        validation = getattr(self, "validation", None)
        if validation is not None:
            validation.close()
        if hasattr(self, "io"):
            if finalize_writer:
                self.io.close()
            else:
                self.io.close(finalize=False)

    def reduce_training_metrics(self, data: dict[str, Any]) -> None:
        """Average scalar training outputs across distributed ranks in place.

        Structured outputs, timings and memory statistics remain rank-local.
        Validation scalars are reduced by :class:`ValidationManager` and are
        appended only after this method runs.

        Parameters
        ----------
        data : dict
            Current processed batch containing model and loss outputs.
        """
        if (
            not TORCH_AVAILABLE
            or not getattr(self, "distributed", False)
            or self.model is None
            or not self.model.train
            or not torch.distributed.is_initialized()
        ):
            return

        scalar_keys = []
        scalar_values = []
        for key, value in data.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                scalar_keys.append(key)
                scalar_values.append(float(value))
            elif torch.is_tensor(value) and value.dim() == 0:
                scalar_keys.append(key)
                scalar_values.append(float(value.item()))

        if not scalar_keys:
            return

        reduced = torch.tensor(
            scalar_values,
            dtype=torch.float64,
            device=self.model.device,
        )
        torch.distributed.all_reduce(reduced)
        reduced /= torch.distributed.get_world_size()
        for key, value in zip(scalar_keys, reduced.tolist()):
            data[key] = value

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
            if getattr(self.io, "columnar", False):
                self.ana.process_columnar(data)
            else:
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
        stdout: bool = True,
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
        stdout : bool, default True
            Whether to emit the human-readable iteration summary. Checkpoint
            rows may print this summary before their CSV row is finalized.
        """
        # Check that the log manager is not being used before initialization
        if self.log_manager is None:
            raise RuntimeError("The log manager must be initialized before logging.")

        log_row = self.log_manager.append(data, self.watch, iteration, epoch)

        if stdout:
            self.log_stdout(data, tstamp, iteration, epoch, log_row)

    def log_stdout(
        self,
        data: dict[str, Any],
        tstamp: str,
        iteration: int,
        epoch: float | None = None,
        log_row: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit an iteration summary without writing a structured log row."""
        if not self.should_log_stdout(iteration):
            return
        if self.log_manager is None:
            raise RuntimeError("The log manager must be initialized before logging.")
        if log_row is None:
            log_row = self.log_manager.collect(data, self.watch, iteration, epoch)

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
