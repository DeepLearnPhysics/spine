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

import os
import random
import subprocess as sc
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import yaml

from .ana import AnaManager
from .construct import BuildManager
from .geo import GeoManager
from .io import IOManager
from .io.write.csv import CSVWriter
from .math import seed as numba_seed
from .model import ModelManager
from .post import PostManager
from .utils.conditional import TORCH_AVAILABLE
from .utils.logger import logger
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

    def __init__(self, cfg: dict[str, Any], rank: int | None = None) -> None:
        """Initializes the class attributes.

        Parameters
        ----------
        cfg : dict
            Global configuration dictionary
        rank : int, optional
            Rank of the GPU. If not specified, the model will be run on CPU if
            `world_size` is 0 and GPU is `world_size` is > 0.
        """
        # Initialize the timers and the configuration dictionary
        self.watch = StopwatchManager()
        self.watch.initialize("iteration")

        # Process the full configuration dictionary and store it
        base, io, geo, model, build, post, ana = self.process_config(**cfg, rank=rank)

        # Initialize the base driver configuration parameters
        train = self.initialize_base(**base, rank=rank)

        # Initialize the detector geometry singleton once and for all modules
        if geo is not None:
            GeoManager.initialize_or_get(**geo)

        # Initialize the input/output
        self.initialize_io(**io)

        # Initialize the ML model
        self.model = None
        if model is not None:
            if self.loader is None:
                raise ValueError(
                    "The model can only be used in conjunction with a loader."
                )
            self.watch.initialize("model")

            # Check if PyTorch is available for model functionality
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for model functionality. "
                    "Install with: pip install spine[model]"
                )

            self.model = ModelManager(
                **model,
                train=train,
                dtype=self.dtype,
                rank=self.rank,
                distributed=self.distributed,
                iter_per_epoch=self.iter_per_epoch,
            )

        else:
            if train is not None:
                raise ValueError(
                    "Received a train block but there is no model to train."
                )

        self.builder = None
        # Initialize the data representation builder
        if build is not None:
            if self.model is not None and not self.unwrap:
                raise ValueError(
                    "Must unwrap the model output to build representations."
                )
            if self.model is not None and not self.model.to_numpy:
                raise ValueError(
                    "Must cast model output to numpy to build representations."
                )
            self.watch.initialize("build")
            self.builder = BuildManager(**build)

        # Initialize the post-processors
        self.post = None
        if post is not None:
            if self.model is not None and not self.unwrap:
                raise ValueError("Must unwrap the model output to run post-processors.")
            self.watch.initialize("post")
            self.post = PostManager(
                post,
                post_list=self.post_list,
                parent_path=self.parent_path,
            )

        # Initialize the analysis scripts
        self.ana = None
        if ana is not None:
            if self.model is not None and not self.unwrap:
                raise ValueError(
                    "Must unwrap the model output to run analysis scripts."
                )
            self.watch.initialize("ana")
            self.ana = AnaManager(ana, log_dir=self.log_dir, prefix=self.log_prefix)

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
        """Reads the configuration and dumps it to the logger.

        Parameters
        ----------
        io : dict
            I/O configuration dictionary
        base : dict, optional
            Base driver configuration dictionary
        geo : dict, optional
            Geometry configuration dictionary
        model : dict, optional
            Model configuration dictionary
        build : dict, optional
            Representation building configuration dictionary
        post : dict, optional
            Post-processor configuration dictionary
        ana : dict, optional
            Analysis script configuration dictionary
        rank : int, optional
            Rank of the GPU. The model will be run on CPU if `world_size` is not
            specified or 0 and on GPU is `world_size` is > 0.

        Returns
        -------
        dict
            Processed configuration
        """
        # If there is no base configuration, make it empty (will use defaults)
        if base is None:
            base = {}

        # Set the verbosity of the logger
        verbosity = base.get("verbosity", "info")
        logger.setLevel(verbosity.upper())

        # Set GPUs visible to CUDA (function handles torch availability)
        base["world_size"] = set_visible_devices(
            world_size=base.get("world_size", None), gpus=base.get("gpus", None)
        )

        # If the seed is not set for the sampler, randomize it. This is done
        # here to keep a record of the seeds provided to the samplers
        if io is None:
            raise ValueError("The `io` block is always required.")
        if "loader" in io:
            if "sampler" in io["loader"]:
                if isinstance(io["loader"]["sampler"], str):
                    io["loader"]["sampler"] = {"name": io["loader"]["sampler"]}

                if (
                    "seed" not in io["loader"]["sampler"]
                    or io["loader"]["sampler"]["seed"] < 0
                ):
                    io["loader"]["sampler"]["seed"] = int(time.time())

        # If the seed is not set for the training/inference process, randomize it
        if "seed" not in base or base["seed"] < 0:
            base["seed"] = int(time.time())
        else:
            if not isinstance(base["seed"], int):
                raise TypeError(
                    f"The driver seed must be an integer, got: {base['seed']}"
                )

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

        # Log information for the main process only
        if rank is None or rank < 1:
            # Log environment information
            logger.info("Release version: %s\n", __version__)

            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            logger.info("$CUDA_VISIBLE_DEVICES=%s\n", visible_devices)

            system_info = sc.getstatusoutput("uname -a")[1]
            logger.info("Configuration processed at: %s\n", system_info)

            # Log configuration
            logger.info(yaml.dump(self.cfg, default_flow_style=None, sort_keys=False))

        # Return updated configuration
        return base, io, geo, model, build, post, ana

    def initialize_base(
        self,
        seed: int,
        dtype: str = "float32",
        world_size: int | None = None,
        gpus: list[int] | None = None,
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
        torch_sharing_strategy: str | None = None,
        split_output: bool = False,
        train: dict[str, Any] | None = None,
        verbosity: str = "info",
    ) -> dict[str, Any] | None:
        """Initialize the base driver parameters.

        Parameters
        ----------
        seed : int
            Random number generator seed
        dtype : str, default 'float32'
            Data type of the model parameters and input data
        world_size : int, optional
            Number of GPUs to use in the underlying model
        gpus : List[int], optional
            List of indexes of GPUs to expose to the model
        log_dir : str, default 'logs'
            Path to the directory where the logs will be written to
        prefix_log : bool, default False
            If True, use the input file name to prefix the log name
        overwrite_log : bool, default False
            If True, overwrite log even if it already exists
        csv_buffer_size : int, default 1
            CSV file buffer size. 1 is line buffered (default, safe),
            -1 uses system default, 0 is unbuffered, >1 is buffer size in bytes
        parent_path : str, optional
            Path to the parent directory of the analysis configuration file
        iterations : int, optional
            Number of entries/batches to process (-1 means all entries)
        epochs : int, optional
            Number of times to iterate over the full dataset
        unwrap : bool, default False
            Wheather to unwrap batched data (only relevant when using loader)
        rank : int, optional
            Rank of the GPU in the multi-GPU training process
        log_step : int, default 1
            Number of iterations before the logging is called (1: every step)
        distributed : bool, default False
            If `True`, this process is distributed among multiple processes
        train : dict, optional
            Training configuration dictionary
        split_output : bool, default False
            Split the output of the process into one file per input file
        verbosity : int, default 'info'
            Verbosity level to pass to the `logging` module. Pick one of
            'debug', 'info', 'warning', 'error', 'critical'.

        Returns
        -------
        dict
            Training configuration
        rank
            Updated rank
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

        return train

    def initialize_io(
        self,
        loader: Mapping[str, Any] | None = None,
        reader: Mapping[str, Any] | None = None,
        writer: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the input/output scripts.

        Parameters
        ----------
        loader : mapping, optional
            PyTorch DataLoader configuration mapping
        reader : mapping, optional
            Reader configuration mapping
        writer : mapping, optional
            Writer configuration mapping
        """
        self.io = IOManager(
            loader=loader,
            reader=reader,
            writer=writer,
            watch=self.watch,
            geo=self.cfg.get("geo"),
            rank=self.rank,
            dtype=self.dtype,
            world_size=self.world_size,
            distributed=self.distributed,
            unwrap=self.unwrap,
            iterations=self.iterations,
            epochs=self.epochs,
            split_output=self.split_output,
        )

        # Expose I/O state on Driver for backwards compatibility.
        self.loader = self.io.loader
        self.loader_iter = self.io.loader_iter
        self.counter = self.io.counter
        self.unwrapper = self.io.unwrapper
        self.reader = self.io.reader
        self.writer = self.io.writer
        self.iter_per_epoch = self.io.iter_per_epoch
        self.post_list = self.io.post_list
        self.log_prefix = self.io.log_prefix
        self.output_prefix = self.io.output_prefix
        self.iterations = self.io.iterations
        self.epochs = self.io.epochs

    def initialize_log(self) -> None:
        """Initialize the output log for this driver process."""
        # Make a directory if it does not exist
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Determine the log name, initialize it
        if self.builder is not None or self.model is None:
            # If running the driver more than a model, give a generic name
            log_name = f"spine_log.csv"
        else:
            # If running the driver within a training/validation process
            # (model only), follow a specific pattern of log names.
            start_iteration = self.model.start_iteration
            prefix = "train" if self.model.train else "inference"
            suffix = "" if not self.model.distributed else f"_proc{self.rank}"
            log_name = f"{prefix}{suffix}_log-{start_iteration:07d}.csv"

        # If requested, prefix the log name with the input file name
        if self.prefix_log:
            suffix = f"_{log_name}"
            log_prefix = self.log_prefix
            if hasattr(self, "io"):
                max_length = max(1, self.io._name_max(self.log_dir) - len(suffix))
                log_prefix = self.io._truncate_prefix(log_prefix, max_length)
            log_name = f"{log_prefix}{suffix}"

        # Initialize the log
        log_path = os.path.join(self.log_dir, log_name)
        self.logger = CSVWriter(
            log_path, overwrite=self.overwrite_log, buffer_size=self.csv_buffer_size
        )

    def __len__(self) -> int:
        """Returns the number of events in the underlying reader object.

        Returns
        -------
        int
            Number of elements in the underlying loader/reader.
        """
        return len(self.reader)

    def __iter__(self) -> "Driver":
        """Resets the counter and returns itself.

        Returns
        -------
        object
            The Driver itself
        """
        # If a loader is used, reinitialize it. Otherwise set an entry counter
        if self.loader is not None:
            self.loader_iter = iter(self.loader)
        else:
            self.counter = 0

        return self

    def __next__(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Defines how to process the next entry in the iterator.

        Returns
        -------
        Union[dict, List[dict]]
            Either one combined data dictionary, or one per entry in the batch
        """
        # If there are more iterations to go through, return data
        if self.loader is not None:
            return self.process()
        else:
            self.counter = 0 if self.counter is None else self.counter
            if self.counter < len(self):
                data = self.process(self.counter)
                self.counter += 1

                return data

            raise StopIteration

    def run(self) -> None:
        """Loop over the requested number of iterations, process them."""
        # To run the loop, must know how many times it must be done
        if self.iterations is None:
            raise ValueError("Must specify either `iterations` or `epochs` parameters.")

        # Initialize the output log
        self.initialize_log()

        # Get the iteration start (if model exists)
        start_iteration = 0
        if self.model is not None and self.model.train:
            start_iteration = self.model.start_iteration

        # Loop and process each iteration
        for iteration in range(start_iteration, self.iterations):
            # When switching to a new epoch, reset the loader iterator
            if self.loader is not None and (
                self.loader_iter is None or iteration % self.iter_per_epoch == 0
            ):
                if self.distributed:
                    epoch_cnt = iteration // self.iter_per_epoch
                    self.loader.sampler.set_epoch(epoch_cnt)
                self.loader_iter = iter(self.loader)

            # Update the epoch counter, record the execution date/time
            epoch = (iteration + 1) / self.iter_per_epoch
            tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process one batch/entry of data
            entry = iteration if self.loader is None else None
            data = self.process(entry=entry, iteration=iteration, epoch=epoch)

            # Log the output
            self.log(data, tstamp, iteration, epoch)

            # Release the memory for the next iteration
            data = None

        # Clean up: close all analysis script CSV writers
        if self.ana is not None:
            self.ana.close()
        if self.writer is not None:
            self.writer.finalize()
            self.writer.close()

    def process(
        self,
        entry: int | None = None,
        run: int | None = None,
        subrun: int | None = None,
        event: int | None = None,
        iteration: int | None = None,
        epoch: float | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
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
        Union[dict, List[dict]]
            Either one combined data dictionary, or one per entry in the batch
        """
        # 0. Make sure there is no watch running, start the iteration timer
        for watch in self.watch.values():
            if watch.running or watch.paused:
                self.watch.reset()
                break

        self.watch.start("iteration")

        # 1. Load data
        data = self.load(entry, run, subrun, event)

        # 2. Pass data through the model
        if self.model is not None:
            self.watch.start("model")
            result = self.model(data, iteration=iteration, epoch=epoch)
            data.update(**result)
            self.watch.stop("model")
            self.watch.update(self.model.watch, "model")

        # 3. Unwrap
        if self.unwrapper is not None:
            self.watch.start("unwrap")
            data = self.unwrapper(data)
            self.watch.stop("unwrap")

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
        if self.writer is not None:
            self.watch.start("write")
            self.writer(data, self.cfg)
            self.watch.stop("write")

        # Stop the iteration timer
        self.watch.stop("iteration")

        # Return
        return data

    def load(
        self,
        entry: int | None = None,
        run: int | None = None,
        subrun: int | None = None,
        event: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Loads one batch/entry to process.

        If the model is run on the fly, the data is batched. Otherwise,
        a single entry is loaded at this stage.

        Parameters
        ----------
        entry : int, optional
            Entry number, only valid with reader
        run : int, optional
            Run number, only valid with reader
        subrun : int, optional
            Subrun number to load
        event : int, optional
            Event number, only valid with reader

        Returns
        -------
        data: dict
            Data dictionary containing the input
        """
        # Dispatch to the appropriate loader
        if self.loader is not None:
            # Can only load batches sequentially, not by index
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

            # Initialize the loader, if necessary
            if self.loader_iter is None:
                self.loader_iter = iter(self.loader)

            # Load the next batch
            self.watch.start("load")
            data = next(self.loader_iter)
            self.watch.stop("load")

        else:
            # Must provide either entry number or both run and event numbers
            if entry is None and (run is None or subrun is None or event is None):
                raise ValueError(
                    "Provide either the entry number or the run, subrun "
                    "and event number to read."
                )

            # Read an entry
            assert (
                self.reader is not None
            )  # For type checker, guaranteed by initialize_io
            self.watch.start("read")
            if entry is not None:
                data = self.reader.get(entry)
            else:
                data = self.reader.get_run_event(run, subrun, event)
            self.watch.stop("read")

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
        # Simply change the underlying entry list
        assert self.reader is not None  # For type checker, guaranteed by initialize_io
        self.reader.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
        )

        # Reset the iterator
        self.loader_iter = None

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
        # Fetch the first entry in the batch
        first_entry = data["index"]
        if isinstance(first_entry, list):
            first_entry = first_entry[0]

        # Fetch the basics
        log_dict = {"iter": iteration, "epoch": epoch, "first_entry": first_entry}

        # Fetch the memory usage (in GB)
        log_dict["cpu_mem"] = psutil.virtual_memory().used / 1.0e9
        log_dict["cpu_mem_perc"] = psutil.virtual_memory().percent
        log_dict["gpu_mem"], log_dict["gpu_mem_perc"] = 0.0, 0.0
        if runtime.cuda_is_available():
            gpu_total = runtime.cuda_mem_info()[-1] / 1.0e9
            log_dict["gpu_mem"] = runtime.cuda_max_memory_allocated() / 1.0e9
            log_dict["gpu_mem_perc"] = 100 * log_dict["gpu_mem"] / gpu_total

        # Fetch the times
        suff = "_time"
        for key, watch in self.watch.items():
            time_iter, time_sum = watch.time, watch.time_sum
            log_dict[f"{key}{suff}"] = time_iter.wall
            log_dict[f"{key}{suff}_cpu"] = time_iter.cpu
            log_dict[f"{key}{suff}_sum"] = time_sum.wall
            log_dict[f"{key}{suff}_sum_cpu"] = time_sum.cpu

        # Fetch all the scalar outputs and append them to a dictionary
        for key in data:
            if np.isscalar(data[key]):
                log_dict[key] = data[key]
            elif runtime.is_tensor(data[key]) and data[key].dim() == 0:
                log_dict[key] = data[key].item()

        # Record
        self.logger.append(log_dict)

        # If requested, log out basics of the training/inference process
        log = ((iteration + 1) % self.log_step) == 0
        if log:
            # Dump general information
            proc = (
                "train" if self.model is not None and self.model.train else "inference"
            )
            device = "GPU" if self.rank is not None else "CPU"
            keys = [f"Time ({proc})", f"{device} memory", "Loss", "Accuracy"]
            widths = [20, 20, 9, 9]
            if self.distributed:
                keys = ["Rank"] + keys
                widths = [5] + widths
            if self.main_process:
                header = "  | " + "| ".join(
                    [f"{keys[i]:<{widths[i]}}" for i in range(len(keys))]
                )
                separator = "  |" + "+".join(["-" * (w + 1) for w in widths])
                msg = f"Iter. {iteration} (epoch {epoch:.3f}) @ {tstamp}\n"
                msg += header + "|\n"
                msg += separator + "|"
                logger.info(msg)
            if self.distributed:
                runtime.distributed_barrier()

            # Dump information pertaining to a specific process
            t_iter = self.watch.time("iteration").wall
            t_net = 0.0
            if self.model is not None:
                t_net = self.watch.time("model").wall

            if self.rank is not None:
                mem, mem_perc = log_dict["gpu_mem"], log_dict["gpu_mem_perc"]
            else:
                mem, mem_perc = log_dict["cpu_mem"], log_dict["cpu_mem_perc"]

            acc = data.get("accuracy", -1.0)
            loss = data.get("loss", -1.0)

            values = [
                f"{t_iter:0.2f} s ({100*t_net/t_iter:0.2f} %)",
                f"{mem:0.2f} GB ({mem_perc:0.2f} %)",
                f"{loss:0.3f}",
                f"{acc:0.3f}",
            ]
            if self.distributed:
                values = [f"{self.rank}"] + values

            msg = "  | " + "| ".join(
                [f"{values[i]:<{widths[i]}}" for i in range(len(keys))]
            )
            msg += "|"
            logger.info(msg)

            # Start new line once only
            if self.distributed:
                runtime.distributed_barrier()
            if self.main_process:
                logger.info("")
