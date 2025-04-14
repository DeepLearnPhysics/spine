"""Basic utilities related to torch's use of CUDA devices."""

import os

import torch


def set_visible_devices(gpus=None, world_size=None, **kwargs):
    """Sets the number of visible CUDA devices based on the base configuration.

    Parameters
    ----------
    gpus : List[int], optional
        List of indexes of GPUs to expose to the model
    world_size : int, optional
        Number of GPUs to use in the model
    **kwargs : dict, optional
        Additional arguments passed to the base configuration

    Returns
    -------
    int
        World size
    """
    # Fetch the list of devices to expose, set the world size accordingly
    if world_size is None and gpus is None:
        # If no GPU request is put in, run on CPU
        world_size = 0
    else:
        # Otherwise, harmonize
        assert ((world_size is None) or (gpus is None) or
                len(gpus) == world_size), (
                        f"The world size ({world_size}) does not match the "
                        f"number of exposed GPUs ({len(gpus)}).")
        world_size = world_size if world_size is not None else len(gpus)
        gpus = gpus if gpus is not None else list(range(world_size))

    # Set the visible CUDA devices
    if not os.environ.get('CUDA_VISIBLE_DEVICES', None) and gpus is not None:
        # If it is not yet set, do it
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(g) for g in gpus])

    # Make sure the world size is consistent with the number of visible GPUs
    if world_size > 0:
        assert torch.cuda.is_available, (
                "Cannot use distributed training without access to GPUs.")

        visible_devices = torch.cuda.device_count()
        assert world_size <= visible_devices, (
                 f"The number of GPUs requested ({world_size}) exceeds the "
                 f"number of visible devices ({visible_devices}).")

    # Return the world size
    return world_size
