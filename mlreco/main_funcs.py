"""Main functions.

This is the first module called when launching a binary script under the ``bin`
directory. It takes care of setting up the environment and the `Driver`
object(s) used to execute/train ML models, post-processors, analysis
scripts, writers and profilers.
"""

import os
import time
import glob
import subprocess as sc

import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from .utils.logger import logger
from .utils.ascii_logo import ascii_logo

from .version import __version__
from .driver import Driver


def run(cfg):
    """Execute a model in one or more processes.

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    """
    # Process the configuration
    cfg, distributed, world_size = process_config(**cfg)

    # Launch the training/inference process
    if not distributed:
        # Run a single process
        run_single(cfg)

    else:
        # Make sure that this is a training process
        assert 'train' in cfg['base'], (
                "Must only used distributed execution for training processes.")

        # Make sure the world size is consistent with the number of visible GPUs
        assert torch.cuda.is_available, (
                "Cannot run a distributed training without access to GPUs.")

        visible_devices = torch.cuda.device_count()
        assert world_size <= visible_devices, (
                 "The number of GPUs requested for distributed execution "
                f"({world_size}) is smaller than the number of visible devices "
                f"({visible_devices}).")

        # Launch the distributed training process
        torch.multiprocessing.spawn(
                train_single, args=(cfg, distributed, world_size), 
                nprocs=world_size)


def run_single(cfg):
    """Execute a model on a single process.

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    """
    # Dispatch
    if 'train' in cfg['base']:
        train_single(cfg=cfg, rank=None)
    else:
        inference_single(cfg)


def train_single(rank, cfg, distributed=False, world_size=None):
    """Train a model in a single process.

    Parameters
    ----------
    rank : int
        Process rank
    cfg : dict
        Full driver/trainer configuration
    distirbuted : bool, default False
        If `True`, distribute the training process
    world_size : int, optional
        Number of devices to use in the distributed training process
    """
    # If distributed, setup the process group
    if distributed:
        setup_ddp(rank, world_size)

    # Prepare the trainer
    driver = Driver(**cfg, rank=rank)

    # Run the training process
    driver.run()


def inference_single(cfg):
    """
    Execute a model in inference mode in a single process

    Parameters
    ----------
    cfg : dict
        Full driver configuration
    """
    # Prepare the driver
    driver = Driver(**cfg)

    # Find the set of weights to run the inference on
    preloaded, weights = False, []
    if driver.model is not None and driver.model.weight_path is not None:
        preloaded = os.path.isfile(driver.model.weight_path)
        weights = sorted(glob.glob(driver.model.weight_path))
        if not preloaded and len(weights):
            weight_list = "\n".join(weights)
            logger.info("Looping over %d set of weights:\n"
                        "%s", len(weights), weight_list)
    if not weights:
        weights = [None]

    # Loop over the weights, run the inference loop
    for weight in weights:
        if weight is not None and not preloaded:
            driver.model.load_weights(weight)
            driver.initialize_log()

        driver.run()


def process_config(base, io, model=None, build=None, post=None,
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

    # Set GPUs visible to CUDA
    world_size = base.get('world_size', 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = \
            ','.join([str(i) for i in range(world_size)])

    # If there is more than one GPU in use, must distribute
    if world_size > 1:
        base['distributed'] = True
    elif 'distributed' not in base:
        base['distributed'] = False

    distributed = base['distributed']

    # If the seed is not set for the sampler, randomize it. This is done
    # here to keep a record of the seeds provided to the samplers
    if 'loader' in io:
        if 'sampler' in io['loader']:
            if isinstance(io['loader']['sampler'], str):
                io['loader']['sampler'] = {'name': io['loader']['sampler']}

            if ('seed' not in io['loader']['sampler'] or
                io['loader']['sampler']['seed'] < 0):
                current = int(time.time())
                if not distributed:
                    io['loader']['sampler']['seed'] = current
                else:
                    io['loader']['sampler']['seed'] = [
                            current + i for i in range(world_size)]

            elif distributed:
                seed = int(io['loader']['sampler']['seed'])
                io['loader']['sampler']['seed'] = [
                        seed + i for i in range(world_size)]

    # If the seed is not set for the training/inference process, randomize it
    if 'seed' not in base or base['seed'] < 0:
        base['seed'] = int(time.time())
    else:
        base['seed'] = int(base['seed'])

    # Set the seed of random number generators
    np.random.seed(base['seed'])
    torch.manual_seed(base['seed'])

    # Rebuild global configuration dictionary
    cfg = {'base': base, 'io': io}
    if model is not None:
        cfg['model'] = model
    if build is not None:
        cfg['build'] = build
    if post is not None:
        cfg['post'] = post
    if ana is not None:
        cfg['ana'] = ana

    # Log package logo
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
    return cfg, distributed, world_size


def apply_event_filter(driver, entry_list):
    """Restrict the list of entries.

    Parameters
    ----------
    driver : Driver
        Driver instance
    n_entry : int, optional
        Maximum number of entries to load
    n_skip : int, optional
        Number of entries to skip at the beginning
    entry_list : list, optional
        List of integer entry IDs to add to the index
    skip_entry_list : list, optional
        List of integer entry IDs to skip from the index
    run_event_list: list((int, int)), optional
        List of [run, event] pairs to add to the index
    skip_run_event_list: list((int, int)), optional
        List of [run, event] pairs to skip from the index
    """
    # Simply change the underlying entry list
    driver.reader.process_entry_list(
            n_entry, n_skip, entry_list, skip_entry_list,
            run_event_list, skip_run_event_list)


def setup_ddp(rank, world_size, backend='nccl'):
    """Sets up the DistributedDataParallel environment."""
    # Define the environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group for this GPU
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
