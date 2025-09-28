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
import subprocess as sc
import time
from datetime import datetime

import numpy as np
import psutil
import yaml

from .ana import AnaManager
from .banner import ascii_logo
from .construct import BuildManager
from .io import reader_factory, writer_factory
from .io.core.write.csv import CSVWriter
from .math import seed as numba_seed
from .model import ModelManager
from .post import PostManager
from .utils.conditional import TORCH_AVAILABLE
from .utils.logger import logger
from .utils.stopwatch import StopwatchManager
from .utils.torch import runtime
from .utils.torch.devices import set_visible_devices
from .utils.unwrap import Unwrapper
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

    def __init__(self, cfg, rank=None):
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
        base, io, model, build, post, ana = self.process_config(**cfg, rank=rank)

        # Initialize the base driver configuration parameters
        train = self.initialize_base(**base, rank=rank)

        # Initialize the input/output
        self.initialize_io(**io)

        # Initialize the ML model
        self.model = None
        if model is not None:
            assert (
                self.loader is not None
            ), "The model can only be used in conjunction with a loader."
            self.watch.initialize("model")

            # Check if PyTorch is available for model functionality
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for model functionality. "
                    "Install with: pip install spine-ml[model]"
                )

            self.model = ModelManager(
                **model,
                train=train,
                dtype=self.dtype,
                rank=self.rank,
                distributed=self.distributed,
            )

        else:
            assert (
                train is None
            ), "Received a train block but there is no model to train."

        # Initialize the data representation builder
        self.builder = None
        if build is not None:
            assert (
                self.model is None or self.unwrap
            ), "Must unwrap the model output to build representations."
            assert (
                self.model is None or self.model.to_numpy
            ), "Must cast model output to numpy to build representations."
            self.watch.initialize("build")
            self.builder = BuildManager(**build)

        # Initialize the post-processors
        self.post = None
        if post is not None:
            assert (
                self.model is None or self.unwrap
            ), "Must unwrap the model output to run post-processors."
            self.watch.initialize("post")
            self.post = PostManager(
                post, post_list=self.post_list, parent_path=self.parent_path
            )

        # Initialize the analysis scripts
        self.ana = None
        if ana is not None:
            assert (
                self.model is None or self.unwrap
            ), "Must unwrap the model output to run analysis scripts."
            self.watch.initialize("ana")
            self.ana = AnaManager(ana, log_dir=self.log_dir, prefix=self.log_prefix)

    def process_config(
        self, io, base=None, model=None, build=None, post=None, ana=None, rank=None
    ):
        """Reads the configuration and dumps it to the logger.

        Parameters
        ----------
        io : dict
            I/O configuration dictionary
        base : dict, optional
            Base driver configuration dictionary
        model : dict, optional
            Model configuration dictionary
        build : dict, optional
            Representation building configuration dictionary
        post : dict, optional
            Post-processor configutation dictionary
        ana : dict, optional
            Analysis script configurationdictionary
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
        base["world_size"] = set_visible_devices(**base)

        # If the seed is not set for the sampler, randomize it. This is done
        # here to keep a record of the seeds provided to the samplers
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
            assert isinstance(
                base["seed"], int
            ), f"The driver seed must be an integer, got: {base['seed']}"

        # Rebuild global configuration dictionary
        self.cfg = {"base": base, "io": io}
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
            # Log package logo
            logger.info(f"\n%s", ascii_logo)

            # Log environment information
            logger.info("Release version: %s\n", __version__)

            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            logger.info("$CUDA_VISIBLE_DEVICES=%s\n", visible_devices)

            system_info = sc.getstatusoutput("uname -a")[1]
            logger.info("Configuration processed at: %s\n", system_info)

            # Log configuration
            logger.info(yaml.dump(self.cfg, default_flow_style=None, sort_keys=False))

        # Return updated configuration
        return base, io, model, build, post, ana

    def initialize_base(
        self,
        seed,
        dtype="float32",
        world_size=None,
        gpus=None,
        log_dir="logs",
        prefix_log=False,
        overwrite_log=False,
        parent_path=None,
        iterations=None,
        epochs=None,
        unwrap=False,
        rank=None,
        log_step=1,
        distributed=False,
        split_output=False,
        train=None,
        verbosity="info",
    ):
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
        np.random.seed(seed)
        numba_seed(seed)
        runtime.manual_seed(seed)

        # Set up the device the model will run on
        if rank is None and world_size > 0:
            assert (
                world_size < 2
            ), "Must not request > 1 GPU without specifying a GPU rank."
            rank = 0

        self.rank = rank
        self.world_size = world_size
        self.main_process = rank is None or rank == 0

        # Check on the distributed process
        assert self.rank is None or self.rank < world_size, (
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
        self.parent_path = parent_path
        self.iterations = iterations
        self.epochs = epochs
        self.unwrap = unwrap
        self.seed = seed
        self.log_step = log_step
        self.split_output = split_output

        return train

    def initialize_io(self, loader=None, reader=None, writer=None):
        """Initializes the input/output scripts.

        Parameters
        ----------
        loader : dict, optional
            PyTorch DataLoader configuration dictionary
        reader : dict, optional
            Reader configuration dictionary
        writer : dict, optional
            Writer configuration dictionary
        """
        # Make sure that we have either a data loader or a reader, not both
        assert (loader is not None) ^ (
            reader is not None
        ), "Must provide either a loader or a reader configuration."

        # Initialize the data loader/reader
        self.loader = None
        self.unwrapper = None
        if loader is not None:
            # Initialize the torch data loader - requires PyTorch
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for loader functionality. "
                    "Install with: pip install spine-ml[model]"
                )

            # Import loader_factory only when PyTorch is available
            from .io import loader_factory

            self.watch.initialize("load")
            self.loader = loader_factory(
                **loader,
                rank=self.rank,
                dtype=self.dtype,
                world_size=self.world_size,
                distributed=self.distributed,
            )

            self.loader_iter = None
            self.iter_per_epoch = len(self.loader)
            self.reader = self.loader.dataset.reader

            # If requested, initialize the unwrapper
            if self.unwrap:
                geo = None
                if hasattr(self.loader, "collate_fn") and hasattr(
                    self.loader.collate_fn, "geo"
                ):
                    geo = self.loader.collate_fn.geo

                self.watch.initialize("unwrap")
                self.unwrapper = Unwrapper(geometry=geo)

            # If working from LArCV files, no post-processor was yet run
            self.post_list = ()

        else:
            # Initialize the reader
            self.watch.initialize("read")
            self.reader = reader_factory(reader)
            self.iter_per_epoch = len(self.reader)

            # Fetch the list of previously run post-processors
            # TODO: this only works with two runs in a row, not 3 and above
            self.post_list = None
            if self.reader.cfg is not None and "post" in self.reader.cfg:
                self.post_list = tuple(self.reader.cfg["post"])

        # Fetch an appropriate common prefix for all input files
        self.log_prefix, self.output_prefix = self.get_prefixes(
            self.reader.file_paths, self.split_output
        )

        # Initialize the data writer, if provided
        self.writer = None
        if writer is not None:
            assert (
                self.loader is None or self.unwrap
            ), "Must unwrap the model output to write it to file."
            self.watch.initialize("write")
            self.writer = writer_factory(
                writer, prefix=self.output_prefix, split=self.split_output
            )

        # Harmonize the iterations and epochs parameters
        assert (self.iterations is None) or (
            self.epochs is None
        ), "Must not specify both `iterations` or `epochs` parameters."
        if self.iterations is not None:
            if self.iterations < 0:
                self.iterations = self.iter_per_epoch
            self.epochs = 1.0
        elif self.epochs is not None:
            self.iterations = int(self.epochs * self.iter_per_epoch)

    @staticmethod
    def get_prefixes(file_paths, split_output):
        """Builds an appropriate output prefix based on the list of input files.

        Parameters
        ----------
        file_paths : List[str]
            List of input file paths
        split_output : bool
            Split the output of the process into one file per input file

        Returns
        -------
        Union[str, List[str]]
            Shared input summary string to be used to prefix outputs
        """
        # Fetch file base names (ignore where they live)
        file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]

        # Get the shared prefix of all files in the list
        prefix = os.path.commonprefix(file_names)

        # If there is only one file, done
        if len(file_names) == 1:
            if not split_output:
                return prefix, prefix
            else:
                return prefix, [prefix]

        # Otherwise, assemble log name from input file names
        sep = "--"
        log_prefix = ""
        if len(prefix):
            log_prefix += prefix

        # Get the shared suffix of all files in the list
        file_names_f = [f[::-1] for f in file_names]
        suffix = os.path.commonprefix(file_names_f)[::-1]
        if prefix == suffix:
            suffix = ""

        # Pad the center of the log name with compnents which are not shared
        first = file_names[0][len(prefix) : len(file_names[0]) - len(suffix)]
        if len(first):
            if len(log_prefix):
                log_prefix += sep
            log_prefix += first

        skip_count = len(file_names) - 2
        if len(file_names) > 2:
            if len(log_prefix):
                log_prefix += sep
            log_prefix += f"{skip_count}"

        last = file_names[-1][len(prefix) : len(file_names[-1]) - len(suffix)]
        if len(last):
            if len(log_prefix):
                log_prefix += sep
            log_prefix += last

        # Add the shared suffix
        if len(suffix):
            log_prefix += f"--{suffix}"

        # Truncate file names that are too long
        max_length = 150
        if len(log_prefix) > max_length:
            log_prefix = log_prefix[: max_length - 3] + "---"

        # Always provide a single prefix for the log, adapt output prefix
        if not split_output:
            return log_prefix, log_prefix
        else:
            return log_prefix, file_names

    def initialize_log(self):
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
            log_name = f"{self.log_prefix}_{log_name}"

        # Initialize the log
        log_path = os.path.join(self.log_dir, log_name)
        self.logger = CSVWriter(log_path, overwrite=self.overwrite_log)

    def __len__(self):
        """Returns the number of events in the underlying reader object.

        Returns
        -------
        int
            Number of elements in the underlying loader/reader.
        """
        return len(self.reader)

    def __iter__(self):
        """Resets the counter and returns itself.

        Returns
        -------
        object
            The Driver itself
        """
        # If a loader is used, reinitialize it. Otherwise set an entry counter
        if self.loader is not None:
            self.loader_iter = iter(self.loader)
            self.counter = None
        else:
            self.counter = 0

        return self

    def __next__(self):
        """Defines how to process the next entry in the iterator.

        Returns
        -------
        Union[dict, List[dict]]
            Either one combined data dictionary, or one per entry in the batch
        """
        # If there are more iterations to go through, return data
        if self.counter < len(self):
            data = self.process(self.counter)
            if self.counter is not None:
                self.counter += 1

            return data

        raise StopIteration

    def run(self):
        """Loop over the requested number of iterations, process them."""
        # To run the loop, must know how many times it must be done
        assert (
            self.iterations is not None
        ), "Must specify either `iterations` or `epochs` parameters."

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
            data = self.process(entry=entry, iteration=iteration)

            # Log the output
            self.log(data, tstamp, iteration, epoch)

            # Release the memory for the next iteration
            data = None

    def process(self, entry=None, run=None, subrun=None, event=None, iteration=None):
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
            result = self.model(data, iteration=iteration)
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

    def load(self, entry=None, run=None, subrun=None, event=None):
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
            assert entry is None and run is None and subrun is None and event is None, (
                "When calling the loader, there is no way to request a "
                "specific entry or run/subrun/event triplet."
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
            assert (entry is not None) or (
                run is not None and subrun is not None and event is not None
            ), (
                "Provide either the entry number or the run, subrun "
                "and event number to read."
            )

            # Read an entry
            self.watch.start("read")
            if entry is not None:
                data = self.reader.get(entry)
            else:
                data = self.reader.get_run_event(run, subrun, event)
            self.watch.stop("read")

        return data

    def apply_filter(
        self,
        n_entry=None,
        n_skip=None,
        entry_list=None,
        skip_entry_list=None,
        run_event_list=None,
        skip_run_event_list=None,
    ):
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

    def log(self, data, tstamp, iteration, epoch=None):
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
            time, time_sum = watch.time, watch.time_sum
            log_dict[f"{key}{suff}"] = time.wall
            log_dict[f"{key}{suff}_cpu"] = time.cpu
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
