"""Main functions.

This is the first module caleld when launching a binary script under the
`bin` directory. It takes care setting up the environment and the
`TrainVal` object(s) used to train/validate ML models.
"""

import os
import time
import glob
import subprocess as sc
from dataclasses import dataclass

import yaml
import torch

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from .iotools.factories import loader_factory

from .utils.torch_local import cycle
from .utils.logger import logger

from .version import __version__

from .trainval import TrainVal


@dataclass
class Handlers:
    """Simple dataclass that holds all the necessary information
    to run the training.
    """
    cfg: dict           = None
    data_io: DataLoader = None
    data_io_iter: iter  = None
    trainval: TrainVal  = None
    #analyzer: Analyzer  = None

    def keys(self):
        """Function which return the available attributes."""
        return list(self.__dict__.keys())


def run(cfg):
    """Execute a model in one or more processes.

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    """
    # Process the configuration
    cfg = process_config(**cfg)

    # Check if we are running in training or inference mode
    assert 'train' in cfg['trainval'], (
            "Must specify the `train` parameter in the `trainval` block")
    train = cfg['trainval']['train']

    # Launch the training/inference process
    if (not 'distributed' in cfg['trainval'] or
        not cfg['trainval']['distributed']):
        run_single(0, cfg, train)
    else:
        world_size = torch.cuda.device_count() \
                if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(run_single,
                args = (cfg, train, world_size,), nprocs = world_size)


def run_single(rank, cfg, train, world_size=None):
    """Execute a model on a single process.

    Parameters
    ----------
    rank : int
        Process rank
    cfg : dict
        IO, model and training/inference configuration
    train : bool
        Whether to train the network or simply execute it
    world_size : int, optional
        How many processes are being run in parallel
    """
    # Treat distributed and undistributed training/inference differently
    if world_size is None:
        if train:
            train_single(cfg)
        else:
            inference_single(cfg)

    else:
        setup_ddp(rank, world_size)
        if train:
            train_single(cfg, rank)
        else:
            inference_single(cfg, rank)
        destroy_process_group()


def train_single(cfg, rank=0):
    """
    Train a model in a single process

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    rank : int, default 0
        Process rank
    """
    # Prepare the handlers
    handlers = prepare(cfg, rank)

    # Run the training process
    handlers.trainval.run()


def inference_single(cfg, rank=0):
    """
    Execute a model in inference mode in a single process

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    rank : int, default 0
        Process rank
    """
    # Prepare the handlers
    handlers = prepare(cfg, rank)

    # Find the set of weights to run the inference on
    preloaded, weights = False, []
    if handlers.trainval.model_path is not None:
        preloaded = os.path.isfile(handlers.trainval.model_path)
        weights = sorted(glob.glob(handlers.trainval.model_path))
        if not preloaded and len(weights):
            weight_list = "\n".join(weights)
            logger.info("Looping over %d set of weights:\n"
                        "%s", len(weights), weight_list)
    if not weights:
        weights = [None]

    # Loop over the weights, run the inference loop
    for weight in weights:
        if weight is not None and not preloaded:
            handlers.trainval.load_weights(weight)
            handlers.trainval.make_directories()

        handlers.trainval.run()


def prepare(cfg, rank=0):
    """Prepares high level API handlers, namely the torch DataLoader (and an
    iterator) and a TrainVal instance.

    Parameters
    ----------
    cfg : dict
        IO, model and training/infernece configuration
    rank : int, default 0
        Process rank

    Returns
    -------
    Handlers
        Handler instances needed for training/inference
    """
    # Initialize the handlers
    handlers = Handlers()

    # If the configuration has not has been processed, do it
    if not cfg.get('processed', False):
        cfg = process_config(**cfg)
    cfg.pop('processed', None)

    # If there is no `trainval` block, treat config as data loading config.
    # Otherwise, intiliaze the train/validation class
    if 'trainval' not in cfg or cfg['trainval'] is None:
        # Instantiate the data loader
        handlers.data_io = loader_factory(**cfg['iotool'])

        # Instantiate a cyclic iterator over the dataloader
        handlers.data_io_iter = iter(cycle(handlers.data_io))

        # Store the configuration dictionary
        handlers.cfg = cfg

    else:
        # Instantiate the training/inference object
        handlers.trainval = TrainVal(**cfg, rank=rank)

        # Expose the dataloader
        handlers.data_io = handlers.trainval.loader

        # Instantiate a cyclic iterator over the dataloader
        handlers.data_io_iter = handlers.trainval.loader_iter

        # Store the configuration dictionary
        handlers.cfg = cfg

    return handlers


def process_config(iotool, model=None, trainval=None, verbosity='info',
                   processed=None):
    """Do all the necessary cross-checks to ensure the that the configuration
    can be used.
    
    Parse the necessary arguments to make them useable downstream.

    Parameters
    ----------
    iotool : dict
        I/O configuration dictionary
    model : dict, optional
        Model configuration dictionary
    trainval : dict, optional
        Training/inference configuration dictionary
    verbosity : int, default 'INFO'
        Verbosity level to pass to the `logging` module. Pick one of
        'debug', 'info', 'warning', 'error', 'critical'.
    processed : bool, default False
        Whether this configuration has already been seen by this function

    Returns
    -------
    cfg : dict
        Complete, updated configuration dictionary
    """
    # If this configuration has already been processed, throw
    if processed is not None and processed:
        raise RuntimeError("Should not process a configuration twice")

    # Set the verbosity of the logger
    if isinstance(verbosity, str):
        verbosity = verbosity.upper()
    logger.setLevel(verbosity)

    # Convert the gpus parameter to a list of GPU IDs
    distributed = False
    world_size = 1
    if trainval is not None and 'gpus' in trainval:
        # Check that the list of GPUs is a comma-separated string
        for val in trainval['gpus'].split(','):
            assert not val or val.isdigit(), (
                    "The `gpus` parameter must be specified as a "
                    "comma-separated string of numbers. "
                   f"Instead, got: {trainval['gpus']}")

        # Set GPUs visible to CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = trainval['gpus']

        # Convert the comma-separated string to a list
        if len(trainval['gpus']):
            trainval['gpus'] = list(range(len(trainval['gpus'].split(','))))
            world_size = len(trainval['gpus'])

        else:
            trainval['gpus'] = []

        # If there is more than one GPU, must distribute
        if world_size > 1:
            trainval['distributed'] = True

        elif 'distributed' not in trainval:
            trainval['distributed'] = False
        distributed = trainval['distributed']

    # If the seed is not set for the sampler, randomize it
    if 'sampler' in iotool:
        if 'seed' not in iotool['sampler'] or iotool['sampler']['seed'] < 0:
            current = int(time.time())
            if not distributed:
                iotool['sampler']['seed'] = current

            else:
                iotool['sampler']['seed'] = \
                        [current + i for i in range(world_size)]
        else:
            if distributed:
                seed = int(iotool['sampler']['seed'])
                iotool['sampler']['seed'] = \
                        [seed + i for i in range(world_size)]

    # If the seed is not set for the training/inference process, randomize it
    if trainval is not None:
        if 'seed' not in trainval or trainval['seed'] < 0:
            trainval['seed'] = int(time.time())

        else:
            trainval['seed'] = int(trainval['seed'])

    # Rebuild global configuration dictionary
    cfg = {'iotool': iotool, 'model': model, 'trainval': trainval}

    # Log package logo
    ascii_logo = (
    " ██████████   ███████████   ███   ███       ██   ███████████\n"
    "███        █  ██       ███   |    ██████    ██   ██         \n"
    "  ████████    ██       ███  ███   ██   ███  ██   ██████████ \n"
    "█        ███  ██████████     |    ██     █████   ██         \n"
    " ██████████   ██            ███   ██       ███   ███████████\n")
    logger.info(f"\n%s", ascii_logo)

    # Log environment information
    logger.info("Release version: %s\n", __version__)

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    logger.info("$CUDA_VISIBLE_DEVICES=%s\n", visible_devices)

    system_info = sc.getstatusoutput('uname -a')[1]
    logger.info("Configuration processed at: %s\n", system_info)

    # Log configuration
    logger.info(yaml.dump(cfg, default_flow_style=None))

    # Add an indication that this configuration has been processed
    cfg['processed'] = True

    # Return updated configuration
    return cfg


def apply_event_filter(handlers, entry_list):
    """Restrict the list of entries

    Parameters
    ----------
    handlers : Handlers
        Handler instances needed for training/inference
    entry_list : list
        List of events to load
    """
    # Simply change the underlying entry list
    handlers.data_io.dataset.reader.process_entry_list(entry_list=entry_list)


def setup_ddp(rank, world_size):
    """Sets up the DistributedDataParallel environment."""
    # Define the environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group for this GPU
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
