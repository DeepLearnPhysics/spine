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
      2. Run the model forward (including loss)
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
        rank : int, default 0
           Rank of the GPU in the multi-GPU training process
        """
        # Initialize the timers and the configuration dictionary
        self.cfg = {} # Move processor elswhere, store it
        self.watch = StopwatchManager()
        self.watch.initialize('iteration')

        # Initialize the base driver configuration parameters
        self.initialize_base(**base, rank=rank)

        # Initialize the input/output
        self.initialize_io(**io)

        # Initialize the ML model
        self.model = None
        if model is not None:
            assert self.loader is not None, (
                    "The model can only be used in conjunction with a loader.")
            self.watch.initialize('model')
            self.model = Model(**model, rank=rank)

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

    def initialize_base(self, log_dir='logs', prefix_log=False,
                        parent_path=None, iterations=-1, epochs=-1,
                        unwrap=False, rank=None, processed=False,
                        seed=None, report_step=1):
        """Initialize the base driver parameters.

        Parameters
        ----------
        log_dir : str, default 'logs'
            Path to the directory where the logs will be written to
        prefix_log : bool, default False
            If True, use the input file name to prefix the log name
        parent_path : str, optional
            Path to the parent directory of the analysis configuration file
        iterations : int, default -1
            Number of entries/batches to process (-1 means all entries)
        epochs : int, default -1
            Number of times to iterate over the full dataset
        unwrap : bool, default False
            Wheather to unwrap batched data (only relevant when using loader)
        rank : int, optional
            Rank of the GPU in the multi-GPU training process
        processed : bool, default False
            Whether this configuration has been pre-processed (must be true)
        seed : int, optional
            Random number generator seed
        report_step : int, default 1
            Number of iterations before the logging is called (1: every step)
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
        self.rank = rank
        self.main_process = rank is None or rank == 0
        self.seed = seed
        self.report_step = report_step

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
            self.loader = loader_factory(**loader, rank=self.rank)
            self.loader_iter = iter(self.loader)
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
                    "Must unwrap the model output to run post-processors.")
            self.watch.initialize('write')
            self.writer = writer_factory(writer)

        # If requested, extract the name of the input file to prefix logs
        if self.prefix_log:
            assert len(self.reader.file_paths) == 1, (
                    "To prefix log, there should be a single input file name.")
            self.log_prefix = pathlib.Path(self.reader.file_paths[0]).stem

        # Harmonize the iterations and epochs parameters
        assert self.iterations ^ self.epochs, (
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
            start_iteration = self.model.start_iteration if self.model else 0
            prefix  = 'train' if self.model.train else 'inference'
            suffix  = '' if not self.model.distributed else f'_proc{self.rank}'
            logname = f'{prefix}{suffix}_log-{start_iteration:07d}.csv'

        # If requested, prefix the log name with the input file name
        if self.prefix_log:
            logname = f'{self.log_prefix}_{logname}'

        self.logger = CSVWriter(logname)

    def run(self):
        """Loop over the requested number of iterations, process them."""
        # Loop and process each iteration
        for iteration in range(self.iterations):
            # Update the epoch counter, record the execution date/time
            epoch = iteration / self.iter_per_epoch
            tstamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Process one batch/entry of data
            data = self.process(iteration)

            # Log the output
            self.log(data, tstamp, iteration, epoch)

    def process(self, iteration=None, run=None, event=None):
        """Process one entry or a batch of entries.

        Run single step of main SPINE driver. This includes data loading,
        model forwarding, data structure building, post-processing
        and appending desired information to each row of output csv files.

        Parameters
        ----------
        iteration : int, optional
            Iteration number for current step.
        run : int, optional
            Run number
        event : int, optional
            Event number

        Returns
        -------
        Union[dict, List[dict]]
            Either one combined data dictionary, or one per entry in the batch
        """
        # 0. Start the timer for the iteration
        self.watch.start('iteration')

        # 1. Load data
        data = self.load(iteration, run, event)

        # 2. Pass data through the model
        if self.model is not None:
            self.watch.start('model')
            result = self.model(data)
            data.update(**result)
            self.watch.stop('model')

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

    def load(self, iteration=None, run=None, event=None):
        """Loads one batch/entry to process.

        If the model is run on the fly, the data is batched. Otherwise,
        a single entry is loaded at this stage.

        Parameters
        ----------
        iteration : int, optional
            Iteration number, only valid with reader
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
            # Can only load batches by index
            assert ((iteration is not None) and
                    (run is None and event is None)), (
                           "Provide the iteration number only.")

            self.watch.start('load')
            data = next(self.loader_iter)
            self.watch.stop('load')

        else:
            # Must provide either iteration or both run and event numbers
            assert ((iteration is not None) or
                    (run is not None and event is not None)), (
                           "Provide either the iteration number or both the "
                           "run number and the event number to load.")

            self.watch.start('read')
            if iteration is not None:
                data = self.reader.get(iteration)
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
        report_step = ((iteration + 1) % self.report_step) == 0
        if report_step:
            # Dump general information
            proc   = 'Train' if self.model.train else 'Inference'
            device = 'GPU' if self.rank is not None else 'CPU'
            keys   = [f'{proc} time', f'{device} memory', 'Loss', 'Accuracy']
            widths = [20, 20, 9, 9]
            if self.model.distributed:
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
            if self.model.distributed:
                torch.model.distributed.barrier()

            # Dump information pertaining to a specific process
            t_iter = self.watch.time('iteration').wall
            t_net  = self.watch.time('model').wall

            if self.rank is not None:
                mem, mem_perc = log_dict['gpu_mem'], log_dict['gpu_mem_perc']
            else:
                mem, mem_perc = log_dict['cpu_mem'], log_dict['cpu_mem_perc']

            acc  = np.mean(data.get('accuracy', -1.))
            loss = np.mean(data.get('loss',     -1.))

            values = [f'{t_net:0.2f} s ({100*t_net/t_iter:0.2f} %)',
                      f'{mem:0.2f} GB ({mem_perc:0.2f} %)',
                      f'{loss:0.3f}', f'{acc:0.3f}']
            if self.model.distributed:
                values = [f'{self.rank}'] + values

            msg = '  | ' + '| '.join(
                    [f'{values[i]:<{widths[i]}}' for i in range(len(keys))])
            msg += '|'
            print(msg, flush=True)

            # Start new line once only
            if self.model.distributed:
                torch.model.distributed.barrier()
            if self.main_process:
                print('', flush=True)
