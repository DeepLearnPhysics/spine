"""Main functions that call the Driver class.

This is the first module called when launching a binary script under the `bin`
directory. It takes care of setting up the environment and the `Driver`
object(s) used to execute/train ML models, post-processors, analysis
scripts, writers and profilers.
"""

import glob
import os

from .driver import Driver
from .utils.conditional import TORCH_AVAILABLE, torch
from .utils.logger import logger
from .utils.torch.devices import set_visible_devices


def run(cfg):
    """Execute a model in one or more processes.

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    """
    # Process the configuration to set up the driver world
    distributed, world_size = process_world(**cfg)

    # Launch the training/inference process
    if not distributed:
        # Run a single process
        run_single(cfg)

    else:
        # Make sure that this is a training process
        assert (
            "train" in cfg["base"]
        ), "Must only used distributed execution for training processes."

        # Launch the distributed training process
        torch.multiprocessing.spawn(
            train_single, args=(cfg, distributed, world_size), nprocs=world_size
        )


def run_single(cfg):
    """Execute a model on a single process.

    Parameters
    ----------
    cfg : dict
        Full driver/trainer configuration
    """
    # Dispatch
    if "train" in cfg["base"]:
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
    distributed : bool, default False
        If `True`, distribute the training process
    world_size : int, optional
        Number of devices to use in the distributed training process
    """
    # Training always requires torch
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install spine-ml[model]"
        )

    # If distributed, setup the process group
    if distributed:
        setup_ddp(rank, world_size)

    # Prepare the trainer
    driver = Driver(cfg, rank)

    # Run the training process
    driver.run()

    # If distributed, destroy the process group
    if distributed:
        torch.distributed.destroy_process_group()


def inference_single(cfg):
    """
    Execute a model in inference mode in a single process

    Parameters
    ----------
    cfg : dict
        Full driver configuration
    """
    # Prepare the driver
    driver = Driver(cfg)

    # Find the set of weights to run the inference on
    preloaded, weights = False, []
    if driver.model is not None and driver.model.weight_path is not None:
        preloaded = os.path.isfile(driver.model.weight_path)
        weights = sorted(glob.glob(driver.model.weight_path))
        if not preloaded and len(weights):
            weight_list = " - " + "\n - ".join(weights)
            logger.info(
                "Looping over %d set of weights:\n%s", len(weights), weight_list
            )
    if not weights:
        weights = [None]

    # Loop over the weights, run the inference loop
    for weight in weights:
        if weight is not None and not preloaded:
            driver.model.load_weights(weight)
            driver.initialize_log()

        driver.run()


def process_world(base, **kwargs):
    """Check on the number of available GPUs and what has been requested.

    Parameters
    ----------
    base : dict
        Base driver configuration dictionary
    **kwargs : dict
        Other elements of the driver configuration
        Analysis script configurationdictionary

    Returns
    -------
    distributed : bool
        If `True`, distribute the training process
    world_size : int
        Number of devices to use in the distributed training process
    """
    # Set the verbosity of the logger
    verbosity = base.get("verbosity", "info")
    logger.setLevel(verbosity.upper())

    # Parse information about the world size, set visible CUDA devices
    world_size = set_visible_devices(**base)

    # If there is more than one GPU in use, must distribute
    distributed = base.get("distributed", world_size > 1)
    assert (
        world_size < 2 or distributed
    ), "Cannot run process on multiple GPUs without distributing it."

    return distributed, world_size


def setup_ddp(rank, world_size, backend="nccl"):
    """Sets up the DistributedDataParallel environment."""
    # Define the environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group for this GPU
    torch.distributed.init_process_group(
        backend=backend, rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)
