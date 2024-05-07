"""Main functions.

This is the first module called when launching a binary script under the
`bin` directory. It takes care of setting up the environment and the
`Driver`/`Trainer` object(s) used to execute/train ML models, post-processors,
and analysis scripts.
"""

import os
import time
import glob
import subprocess as sc

import yaml
import torch

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from .utils.logger import logger

from .version import __version__

from .driver import Driver
from .train import Trainer


def run(cfg):
    """Execute a model in one or more processes.

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    """
    # Process the configuration
    cfg = process_config(**cfg)

    # Dispatch
    if 'train' not in cfg:
        # If this is not a training process, use the driver
        driver = Driver(**cfg)
        driver.run()

    else:
        # Launch the training/inference process
        if (not 'distributed' in cfg['train'] or
            not cfg['train']['distributed']):
            run_single(0, cfg, train)
        else:
            world_size = 1
            if torch.cuda.is_available:
                world_size = torch.cuda.is_avaialble()
            torch.multiprocessing.spawn(run_single,
                    args = (cfg, train, world_size,), nprocs = world_size)


def run_single(rank, cfg, train, world_size=None):
    """Execute a model on a single process.

    Parameters
    ----------
    rank : int
        Process rank
    cfg : dict
        Full driver/trainer configuration
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
        Full driver/trainer configuration
    rank : int, default 0
        Process rank
    """
    # Prepare the trainer
    trainer = Trainer(**cfg, rank=rank)

    # Run the training process
    trainer.run()


def inference_single(cfg, rank=0):
    """
    Execute a model in inference mode in a single process

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    rank : int, default 0
        Process rank
    """
    # Prepare the trainer
    trainer = Trainer(**cfg, rank=rank)

    # Find the set of weights to run the inference on
    preloaded, weights = False, []
    if trainer.model.model_path is not None:
        preloaded = os.path.isfile(trainer.model.model_path)
        weights = sorted(glob.glob(trainer.model.model_path))
        if not preloaded and len(weights):
            weight_list = "\n".join(weights)
            logger.info("Looping over %d set of weights:\n"
                        "%s", len(weights), weight_list)
    if not weights:
        weights = [None]

    # Loop over the weights, run the inference loop
    for weight in weights:
        if weight is not None and not preloaded:
            trainer.model.load_weights(weight)
            trainer.make_directories()

        trainer.run()


def process_config(base, io, model=None, train=None, build=None, post=None,
                   ana=None, verbosity='info', processed=False):
    """Do all the necessary cross-checks to ensure that the configuration
    can be used.
    
    Parse the necessary arguments to make them useable downstream.

    Parameters
    ----------
    base : dict
        Base driver configuration dictionary
    io : dict
        I/O configuration dictionary
    model : dict, optional
        Model configuration dictionary
    train : dict, optional
        Training configuration dictionary
    build : dict, optional
        Representation building configuration dictionary
    post : dict, optional
        Post-processor configutation dictionary
    ana : dict, optional
        Analysis script configurationdictionary
    verbosity : int, default 'info'
        Verbosity level to pass to the `logging` module. Pick one of
        'debug', 'info', 'warning', 'error', 'critical'.

    Returns
    -------
    cfg : dict
        Complete, updated configuration dictionary
    """
    # If this configuration has already been processed, throw
    if 'processed' in base and base['processed']:
        raise RuntimeError("Must not process a configuration twice.")
    else:
        base['processed'] = True

    # Set the verbosity of the logger
    if isinstance(verbosity, str):
        verbosity = verbosity.upper()
    logger.setLevel(verbosity)

    # Convert the gpus parameter to a list of GPU IDs
    distributed = False
    world_size = 1
    if train is not None and 'gpus' in train:
        # Check that the list of GPUs is a comma-separated string
        for val in train['gpus'].split(','):
            assert not val or val.isdigit(), (
                    "The `gpus` parameter must be specified as a "
                    "comma-separated string of numbers. "
                   f"Instead, got: {train['gpus']}")

        # Set GPUs visible to CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = train['gpus']

        # Convert the comma-separated string to a list
        if len(train['gpus']):
            train['gpus'] = list(range(len(train['gpus'].split(','))))
            world_size = len(train['gpus'])

        else:
            train['gpus'] = []

        # If there is more than one GPU, must distribute
        if world_size > 1:
            train['distributed'] = True

        elif 'distributed' not in train:
            train['distributed'] = False
        distributed = train['distributed']

    # If the seed is not set for the sampler, randomize it. This is done
    # here to keep a record of the seeds provided to the samplers
    if 'loader' in io and 'sampler' in io['loader']:
        if ('seed' not in io['loader']['sampler'] or
            io['loader']['sampler']['seed'] < 0):
            current = int(time.time())
            if not distributed:
                io['loader']['sampler']['seed'] = current
            else:
                io['loader']['sampler']['seed'] = [
                        current + i for i in range(world_size)]

        else:
            if distributed:
                seed = int(io['loader']['sampler']['seed'])
                io['loader']['sampler']['seed'] = [
                        seed + i for i in range(world_size)]

    # If the seed is not set for the training/inference process, randomize it
    if train is not None:
        if 'seed' not in train or train['seed'] < 0:
            train['seed'] = int(time.time())
        else:
            train['seed'] = int(train['seed'])

    # Rebuild global configuration dictionary
    cfg = {'base': base, 'io': io}
    if model is not None:
        cfg['model'] = model
    if train is not None:
        cfg['train'] = train
    if build is not None:
        cfg['build'] = build
    if post is not None:
        cfg['post'] = post
    if ana is not None:
        cfg['ana'] = ana

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
    logger.info(yaml.dump(cfg, default_flow_style=None, sort_keys=False))

    # Return updated configuration
    return cfg


def apply_event_filter(driver, entry_list):
    """Restrict the list of entries

    Parameters
    ----------
    driver : Driver
        Driver instance
    entry_list : list
        List of events to load
    """
    # Simply change the underlying entry list
    driver.reader.process_entry_list(entry_list)


def setup_ddp(rank, world_size):
    """Sets up the DistributedDataParallel environment."""
    # Define the environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group for this GPU
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
