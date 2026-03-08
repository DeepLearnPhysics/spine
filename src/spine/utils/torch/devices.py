"""CUDA device management and distributed training setup.

This module provides utilities for configuring GPU devices and setting up
distributed training environments with graceful PyTorch unavailability handling.
"""

import os
from typing import List, Optional

from ..conditional import TORCH_AVAILABLE, torch

__all__ = ["set_visible_devices"]


def set_visible_devices(
    gpus: Optional[List[int]] = None, world_size: Optional[int] = None
) -> int:
    """Sets the number of visible CUDA devices based on the base configuration.

    Parameters
    ----------
    gpus : List[int], optional
        List of indexes of GPUs to expose to the model
    world_size : int, optional
        Number of GPUs to use in the model

    Returns
    -------
    int
        World size
    """
    # Check if torch is available for GPU operations
    if not TORCH_AVAILABLE:
        if gpus is not None or world_size is not None and world_size > 0:
            raise ImportError(
                "PyTorch is required for GPU operations. "
                "Install with: pip install spine[model]"
            )
        return 0  # CPU only

    # If both gpus and world_size are provided, check for consistency
    if world_size is not None and gpus is not None and len(gpus) != world_size:
        raise ValueError(
            f"The world size ({world_size}) does not match the "
            f"number of exposed GPUs ({len(gpus)})."
        )

    # Fetch the list of devices to expose, set the world size accordingly
    if world_size is None:
        world_size = 0 if gpus is None else len(gpus)
    if gpus is None:
        gpus = [] if world_size == 0 else list(range(world_size))

    # Set the visible CUDA devices
    if not os.environ.get("CUDA_VISIBLE_DEVICES", None) and gpus is not None:
        # If it is not yet set, do it
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])

    # Make sure the world size is consistent with the number of visible GPUs
    # Skip this check in multi-node mode (RANK set) since SLURM/torchrun handles validation
    if world_size > 0 and "RANK" not in os.environ:
        assert (
            torch.cuda.is_available
        ), "Cannot use distributed training without access to GPUs."

        visible_devices = torch.cuda.device_count()
        assert world_size <= visible_devices, (
            f"The number of GPUs requested ({world_size}) exceeds the "
            f"number of visible devices ({visible_devices})."
        )

    # Return the world size
    return world_size
