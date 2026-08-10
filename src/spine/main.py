"""Runtime orchestration for the central :class:`spine.driver.Driver`.

This is the first module called when launching a binary script under the `bin`
directory. It resolves the requested device world, launches one process per
rank when needed, initializes optional DistributedDataParallel (DDP) state and
runs the configured training, inference or analysis workflow.
"""

import os
from typing import Optional, Tuple

from .config import normalize_config
from .driver import Driver
from .utils.conditional import TORCH_AVAILABLE, torch
from .utils.logger import configure_rank_logging, logger
from .utils.torch.devices import set_visible_devices


def run(cfg: dict) -> None:
    """Launch a configured SPINE workflow in one or more processes.

    Single-process workflows are executed directly. Distributed workflows
    either use the rank supplied by an external launcher such as SLURM or
    ``torchrun``, or spawn one local process per requested device.

    Parameters
    ----------
    cfg : dict
        Complete SPINE driver configuration. It must contain a ``base`` block
        describing the execution world.

    Raises
    ------
    ValueError
        If the configuration does not contain a ``base`` block or requests an
        invalid execution world.
    """
    # Normalize legacy block locations once before launching worker processes.
    cfg = normalize_config(cfg)

    # Process the configuration to set up the driver world
    if "base" not in cfg:
        raise ValueError("Configuration must contain a 'base' section.")
    distributed, world_size, torch_sharing = process_world(cfg["base"])

    # Check if rank is provided externally (multi-node/SLURM setup)
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else None

    # Launch the training/inference process
    if not distributed:
        # Run a single process on a single GPU (or CPU if no GPUs available)
        run_single(None, cfg)

    elif rank is not None:
        # Multi-node: rank provided externally by SLURM/torchrun, run directly
        run_single(rank, cfg, distributed, world_size, torch_sharing)

    else:
        # Single-node multi-GPU: launch processes using multiprocessing.spawn
        torch.multiprocessing.spawn(
            run_single,
            args=(cfg, distributed, world_size, torch_sharing),
            nprocs=world_size,
        )


def run_single(
    rank: Optional[int],
    cfg: dict,
    distributed: bool = False,
    world_size: Optional[int] = None,
    torch_sharing: Optional[str] = None,
) -> None:
    """Execute one training or inference worker process.

    This function follows the rank-first calling convention required by
    :func:`torch.multiprocessing.spawn`. It is also used directly for ordinary
    single-process execution and for ranks created by external multi-node
    launchers.

    Parameters
    ----------
    rank : int, optional
        Global process rank. ``None`` identifies a non-distributed process.
    cfg : dict
        Complete SPINE driver configuration.
    distributed : bool, default False
        If ``True``, this worker participates in distributed execution and its
        loader is sharded by ``rank`` and ``world_size``.
    world_size : int, optional
        Total number of processes across all nodes. Required when
        ``distributed`` is ``True``.
    torch_sharing : str or None, optional
        PyTorch multiprocessing file-sharing strategy.

    Raises
    ------
    ImportError
        If PyTorch is unavailable for training or distributed execution.
    ValueError
        If distributed training explicitly disables DDP.
    """
    # Normalize direct worker calls as well as launcher-managed calls.
    cfg = normalize_config(cfg)

    # Determine the execution mode from the presence of a training block
    train = "train" in cfg

    # Validate requirements shared by training and distributed inference
    if (train or distributed) and not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training or distributed execution. "
            "Install with: pip install spine[model]"
        )

    if train and distributed and not cfg["base"].get("ddp", True):
        raise ValueError("Distributed training requires `base.ddp: true`.")

    # Configure rank-aware logging before initializing worker-owned modules
    configure_rank_logging(rank)

    # Initialize the process-local device and optional DDP process group
    if distributed:
        assert rank is not None and world_size is not None
        if torch_sharing is not None:
            torch.multiprocessing.set_sharing_strategy(torch_sharing)

        # Non-DDP inference still needs a rank-local device for independent
        # model execution; data sharding is handled separately by the loader.
        if cfg["base"].get("ddp", True):
            setup_ddp(rank, world_size)
        else:
            set_process_device(rank)

    # Build and execute the driver, then release distributed resources
    try:
        driver = Driver(cfg, rank)
        if train:
            driver.run()
        else:
            run_inference(driver)
    finally:
        if distributed and cfg["base"].get("ddp", True):
            torch.distributed.destroy_process_group()


def run_inference(driver: Driver) -> None:
    """Run a prepared driver for each configured inference checkpoint.

    Parameters
    ----------
    driver : Driver
        Initialized inference driver. Its model may provide one checkpoint, a
        sorted collection of checkpoints or no pretrained weights.
    """
    # Resolve the checkpoint sequence; scalar paths are loaded by the manager
    preloaded, weights = False, []
    if driver.model is not None:
        weights = driver.model.weight_path
        if weights is None or isinstance(weights, str):
            preloaded = True
            weights = [weights]
        else:
            weights = sorted(weights)
            weight_list = " - " + "\n - ".join(weights)
            logger.info(
                "Looping over %d set of weights:\n%s", len(weights), weight_list
            )
    if not weights:
        weights = [None]

    # Run once per checkpoint, reloading only collections handled at this level
    for weight in weights:
        if driver.model is not None and weight is not None and not preloaded:
            driver.model.load_weights(weight)
            driver.initialize_log()

        driver.run()


def set_process_device(rank: int) -> None:
    """Assign a distributed worker to its node-local CUDA device.

    Parameters
    ----------
    rank : int
        Global process rank. It is used as the device index for local spawning
        when an external launcher does not provide ``LOCAL_RANK``.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)


def process_world(base: dict) -> Tuple[bool, int, Optional[str]]:
    """Resolve and validate the requested execution world.

    Parameters
    ----------
    base : dict
        Base driver configuration dictionary.

    Returns
    -------
    distributed : bool
        Whether execution should be distributed across ranks.
    world_size : int
        Number of requested execution processes.
    torch_sharing : str or None
        Validated PyTorch multiprocessing file-sharing strategy.

    Raises
    ------
    ValueError
        If multiple devices are requested with distribution disabled, or if
        the requested file-sharing strategy is invalid.
    """
    # Set the verbosity of the logger
    verbosity = base.get("verbosity", "info")
    logger.setLevel(verbosity.upper())

    # Parse information about the world size, set visible CUDA devices
    world_size = set_visible_devices(
        world_size=base.get("world_size", None), gpus=base.get("gpus", None)
    )

    # If there is more than one GPU in use, must distribute
    distributed = base.get("distributed", world_size > 1)
    if world_size > 1 and not distributed:
        raise ValueError(
            "Multiple GPUs detected but distributed execution is disabled. "
            "Set 'distributed: true' in the configuration to enable it."
        )

    # If distributed, check what the file sharing strategy is
    torch_sharing = base.get("torch_sharing_strategy", None)
    if torch_sharing is not None and torch_sharing not in (
        "file_system",
        "file_descriptor",
    ):
        raise ValueError(
            "torch_sharing_strategy must be one of: "
            "'file_system', 'file_descriptor', or None"
        )

    return distributed, world_size, torch_sharing


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the process group for a DDP worker.

    Parameters
    ----------
    rank : int
        Global rank of this process, in ``[0, world_size)``.
    world_size : int
        Total number of processes across all nodes.
    backend : str, default "nccl"
        PyTorch distributed backend.

    Notes
    -----
    External multi-node launchers should provide ``MASTER_ADDR``,
    ``MASTER_PORT``, ``RANK``, ``WORLD_SIZE`` and, when multiple devices are
    available on each node, ``LOCAL_RANK``. Single-node spawning falls back to
    a local rendezvous address and uses the global rank as the local rank.
    """
    # Set master address and port from environment, or use defaults for single-node
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    # Select the node-local device before initializing the NCCL process group.
    set_process_device(rank)

    # Initialize the process group for this GPU
    torch.distributed.init_process_group(
        backend=backend, rank=rank, world_size=world_size
    )
