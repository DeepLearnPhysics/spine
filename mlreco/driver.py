"""SPINE driver class.

Takes care of everything in one centralized place:
    - Data loading
    - ML model and loss forward pass
    - Batch unwrapping
    - Representation building
    - Post-processing
    - Analysis script execution
    - Writing to file
"""

import os
from datetime import datetime

import psutil
import pathlib
import numpy as np
import torch

from .io import loader_factory, reader_factory, writer_factory
from .io.write import CSVWriter

from .model import Model
from .build import BuildManager
from .post import PostManager
from .ana import AnaManager

from .utils.unwrap import Unwrapper
from .utils.stopwatch import StopwatchManager

__all__ = ['Driver']


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
    """

    def __init__(self, base, io, model=None, build=None, post=None,
                 ana=None, rank=None):
        """Initializes the class attributes.

        Parameters
        ----------
        base : dict
           Base driver configuration dictionary
        io : dict
           Input/output configuration dictionary
        model : dict
           Model configuration dictionary
        post : dict, optional
            Post-processor configuration, if there are any to be run
        ana : dict, optional
            Analysis script configuration (writes to CSV files)
        rank : int, optional
           Rank of the GPU in the multi-GPU training process. If not specified,
           the underlying ML process is run on CPU.
        """
        # Initialize the timers and the configuration dictionary
        self.cfg = {} # Move processor elswhere, store it
        self.watch = StopwatchManager()
        self.watch.initialize('iteration')

        # Initialize the base driver configuration parameters
        train, rank = self.initialize_base(**base, rank=rank)

        # Initialize the input/output
        self.initialize_io(**io)

        # Initialize the ML model
        self.model = None
        if model is not None:
            assert self.loader is not None, (
                    "The model can only be used in conjunction with a loader.")
            self.watch.initialize('model')
            self.model = Model(
                    **model, train=train, rank=rank,
                    distributed=self.distributed)
        else:
            assert train is None, (
                    "Received a train block but there is no model to train.")

        # Initialize the data representation builder
        self.builder = None
        if build is not None:
            assert self.model is None or self.unwrap, (
                    "Must unwrap the model output to build representations.")
            assert self.model is None or self.model.to_numpy, (
                    "Must cast model output to numpy to build representations.")
            self.watch.initialize('build')
            self.builder = BuildManager(**build)

        # Initialize the post-processors
        self.post = None
        if post is not None:
            assert self.model is None or self.unwrap, (
                    "Must unwrap the model output to run post-processors.")
            self.watch.initialize('post')
            self.post = PostManager(post, parent_path=self.parent_path)

        # Initialize the analysis scripts
        self.ana = None
        if ana is not None:
            assert self.model is None or self.unwrap, (
                    "Must unwrap the model output to run analysis scripts.")
            self.watch.initialize('ana')
            self.ana = AnaManager(ana)

        # Initialize the output log
        self.initialize_log()

    def initialize_base(self, world_size=0, log_dir='logs', prefix_log=False,
                        parent_path=None, iterations=None, epochs=None,
                        unwrap=False, rank=None, processed=False,
                        seed=None, log_step=1, distributed=False, train=None):
        """Initialize the base driver parameters.

        Parameters
        ----------
        world_size : int, default 0
            Number of GPUs to use in the underlying model
        log_dir : str, default 'logs'
            Path to the directory where the logs will be written to
        prefix_log : bool, default False
            If True, use the input file name to prefix the log name
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
        processed : bool, default False
            Whether this configuration has been pre-processed (must be true)
        seed : int, optional
            Random number generator seed
        log_step : int, default 1
            Number of iterations before the logging is called (1: every step)
        distributed : bool, default False
            If `True`, this process is distributed among multiple processes

        Returns
        -------
        dict
            Training configuration
        rank
            Updated rank
        """
        # Check that the configuration has been processed
        assert processed, (
                "Must run process_config from spine.main_funcs before "
                "initialializing the full driver class.")

        # Store general parameters
        self.log_dir = log_dir
        self.prefix_log = prefix_log
        self.parent_path = parent_path
        self.iterations = iterations
        self.epochs = epochs
        self.unwrap = unwrap
        self.seed = seed
        self.log_step = log_step

        # Process the process GPU rank
        if rank is None and world_size > 0:
            assert world_size < 2, (
                    "Must not request > 1 GPU without specifying a GPU rank.")
            rank = 0

        self.rank = rank
        self.world_size = world_size
        self.main_process = rank is None or rank == 0

        # Check on the distributed process
        assert self.rank is None or self.rank < world_size, (
                f"The GPU rank index of this driver ({rank}) is too large "
                f"for the number of GPUs available ({world_size}).")

        self.distributed = distributed
        if not distributed and world_size > 1:
            self.distributed = True

        return train, rank

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
        assert (loader is not None) ^ (reader is not None), (
                "Must provide either a loader or a reader configuration.")

        # Initialize the data loader/reader
        self.loader = None
        if loader is not None:
            # Initialize the torch data loader
            self.watch.initialize('load')
            self.loader = loader_factory(
                    **loader, rank=self.rank, world_size=self.world_size,
                    distributed=self.distributed)
            self.loader_iter = None
            self.iter_per_epoch = len(self.loader)
            self.reader = self.loader.dataset.reader

            # If requested, initialize the unwrapper
            if self.unwrap:
                geometry = None
                if (hasattr(self.loader, 'collate_fn') and
                    hasattr(self.loader.collate_fn, 'geo')):
                    geometry = self.loader.collate_fn.geo

                self.watch.initialize('unwrap')
                self.unwrapper = Unwrapper(geometry=geometry)

        else:
            # Initialize the reader
            self.watch.initialize('read')
            self.reader = reader_factory(reader)
            self.iter_per_epoch = len(self.reader)

        # Initialize the data writer, if provided
        self.writer = None
        if writer is not None:
            assert self.loader is None or self.unwrap, (
                    "Must unwrap the model output to write it to file.")
            self.watch.initialize('write')
            self.writer = writer_factory(writer)

        # If requested, extract the name of the input file to prefix logs
        if self.prefix_log:
            assert len(self.reader.file_paths) == 1, (
                    "To prefix log, there should be a single input file name.")
            self.log_prefix = pathlib.Path(self.reader.file_paths[0]).stem

        # Harmonize the iterations and epochs parameters
        assert (self.iterations is not None) ^ (self.epochs is not None), (
                "Must specify either `iterations` or `epochs` parameters.")
        if self.iterations is not None:
            if self.iterations < 0:
                self.iterations = self.iter_per_epoch
            self.epochs = 1.
        else:
            self.iterations = self.epochs*self.iter_per_epoch

    def initialize_log(self):
        """Initialize the output log for this driver process."""
        # Make a directory if it does not exist
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Determine the log name, initialize it
        if self.model is None:
            # If running the driver with a model, give a generic name
            logname = f'spine_log.csv'
        else:
            # If running the driver within a training/validation process,
            # follow a specific pattern of log names.
            start_iteration = self.model.start_iteration
            prefix  = 'train' if self.model.train else 'inference'
            suffix  = '' if not self.model.distributed else f'_proc{self.rank}'
            logname = f'{prefix}{suffix}_log-{start_iteration:07d}.csv'

        # If requested, prefix the log name with the input file name
        if self.prefix_log:
            logname = f'{self.log_prefix}_{logname}'

        self.logger = CSVWriter(os.path.join(self.log_dir, logname))

    def run(self):
        """Loop over the requested number of iterations, process them."""
        # Get the iteration start (if model exists
        start_iteration = 0
        if self.model is not None and self.model.train:
            start_iteration = self.model.start_iteration
        epoch = start_iteration/self.iter_per_epoch

        # Loop and process each iteration
        for iteration in range(start_iteration, self.iterations):
            # When switching to a new epoch, reset the loader iterator
            if (self.loader is not None and
                iteration//self.iter_per_epoch != epoch//1):
                if self.distributed:
                    self.loader.sampler.set_epoch(e)
                self.loader_iter = iter(self.loader)

            # Update the epoch counter, record the execution date/time
            epoch = iteration / self.iter_per_epoch
            tstamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Process one batch/entry of data
            data = self.process(iteration=iteration)

            # Log the output
            self.log(data, tstamp, iteration, epoch)

            # Release the memory for the next iteration
            data = None

    def process(self, entry=None, run=None, event=None, iteration=None):
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
        # 0. Start the timer for the iteration
        self.watch.start('iteration')

        # 1. Load data
        data = self.load(entry, run, event)

        # 2. Pass data through the model
        if self.model is not None:
            self.watch.start('model')
            result = self.model(data, iteration=iteration)
            data.update(**result)
            self.watch.stop('model')
            self.watch.update(self.model.watch, 'model')

        # 3. Unwrap
        if self.unwrap:
            self.watch.start('unwrap')
            data = self.unwrapper(data)
            self.watch.stop('unwrap')

        # 4. Build representations
        if self.builder is not None:
            self.watch.start('build')
            self.builder(data)
            self.watch.stop('build')

        # 5. Run post-processing, if requested
        if self.post is not None:
            self.watch.start('post')
            self.post(data)
            self.watch.stop('post')
            self.watch.update(self.post.watch, 'post')

        # 6. Run scripts, if requested
        if self.ana is not None:
            self.watch.start('ana')
            self.ana(data)
            self.watch.stop('ana')
            self.watch.update(self.post.watch, 'ana')

        # 7. Write output to file, if requested
        if self.writer is not None:
            self.watch.start('write')
            self.writer(data, self.cfg)
            self.watch.stop('write')

        # Stop the iteration timer
        self.watch.stop('iteration')

        # Return
        return data

    def load(self, entry=None, run=None, event=None):
        """Loads one batch/entry to process.

        If the model is run on the fly, the data is batched. Otherwise,
        a single entry is loaded at this stage.

        Parameters
        ----------
        entry : int, optional
            Entry number, only valid with reader
        run : int, optional
            Run number, only valid with reader
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
            assert (entry is None and run is None and event is None), (
                    "When calling the loader, no way to specify a specific "
                    "entry or run/event pair.")

            # Initialize the loader, if necessary
            if self.loader_iter is None:
                self.loader_iter = iter(self.loader)

            # Load the next batch
            self.watch.start('load')
            data = next(self.loader_iter)
            self.watch.stop('load')

        else:
            # Must provide either entry number or both run and event numbers
            assert ((entry is not None) or
                    (run is not None and event is not None)), (
                           "Provide either the entry number or both the "
                           "run number and the event number to read.")

            # Read an entry
            self.watch.start('read')
            if entry is not None:
                data = self.reader.get(entry)
            else:
                data = self.reader.get_run_event(run, event)
            self.watch.stop('read')

        return data

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
        first_entry = data['index']
        if isinstance(first_entry, list):
            first_entry = first_entry[0]

        # Fetch the basics
        log_dict = {
            'iter': iteration,
            'epoch': epoch,
            'first_entry': first_entry
        }

        # Fetch the memory usage (in GB)
        log_dict['cpu_mem'] = psutil.virtual_memory().used/1.e9
        log_dict['cpu_mem_perc'] = psutil.virtual_memory().percent
        log_dict['gpu_mem'], log_dict['gpu_mem_perc'] = 0., 0.
        if torch.cuda.is_available():
            gpu_total = torch.cuda.mem_get_info()[-1] / 1.e9
            log_dict['gpu_mem'] = torch.cuda.max_memory_allocated() / 1.e9
            log_dict['gpu_mem_perc'] = 100 * log_dict['gpu_mem'] / gpu_total

        # Fetch the times
        suff = '_time'
        for key, watch in self.watch.items():
            time, time_sum = watch.time, watch.time_sum
            log_dict[f'{key}{suff}'] = time.wall
            log_dict[f'{key}{suff}_cpu'] = time.cpu
            log_dict[f'{key}{suff}_sum'] = time_sum.wall
            log_dict[f'{key}{suff}_sum_cpu'] = time_sum.cpu

        # Fetch all the scalar outputs and append them to a dictionary
        for key in data:
            if np.isscalar(data[key]):
                log_dict[key] = data[key]

        # Record
        self.logger.append(log_dict)

        # If requested, print out basics of the training/inference process.
        log = ((iteration + 1) % self.log_step) == 0
        if log:
            # Dump general information
            proc   = 'Train' if self.model is not None and self.model.train else 'Inference'
            device = 'GPU' if self.rank is not None else 'CPU'
            keys   = [f'{proc} time', f'{device} memory', 'Loss', 'Accuracy']
            widths = [20, 20, 9, 9]
            if self.distributed:
                keys = ['Rank'] + keys
                widths = [5] + widths
            if self.main_process:
                header = '  | ' + '| '.join(
                        [f'{keys[i]:<{widths[i]}}' for i in range(len(keys))])
                separator = '  |' + '+'.join(['-'*(w+1) for w in widths])
                msg  = f"Iter. {iteration} (epoch {epoch:.3f}) @ {tstamp}\n"
                msg += header + '|\n'
                msg += separator + '|'
                print(msg, flush=True)
            if self.distributed:
                torch.distributed.barrier()

            # Dump information pertaining to a specific process
            t_iter = self.watch.time('iteration').wall
            t_net  = 0.
            if self.model is not None:
                t_net  = self.watch.time('model').wall

            if self.rank is not None:
                mem, mem_perc = log_dict['gpu_mem'], log_dict['gpu_mem_perc']
            else:
                mem, mem_perc = log_dict['cpu_mem'], log_dict['cpu_mem_perc']

            acc  = data.get('accuracy', -1.)
            loss = data.get('loss', -1.)

            values = [f'{t_net:0.2f} s ({100*t_net/t_iter:0.2f} %)',
                      f'{mem:0.2f} GB ({mem_perc:0.2f} %)',
                      f'{loss:0.3f}', f'{acc:0.3f}']
            if self.distributed:
                values = [f'{self.rank}'] + values

            msg = '  | ' + '| '.join(
                    [f'{values[i]:<{widths[i]}}' for i in range(len(keys))])
            msg += '|'
            print(msg, flush=True)

            # Start new line once only
            if self.distributed:
                torch.distributed.barrier()
            if self.main_process:
                print('', flush=True)
