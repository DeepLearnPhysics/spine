"""Functions that instantiate IO tools from configuration blocks."""

from warnings import warn

from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.factory import instantiate, module_dict

from . import read, write

READER_DICT = module_dict(read)
WRITER_DICT = module_dict(write)

__all__ = [
    "reader_factory",
    "writer_factory",
    "loader_factory",
    "dataset_factory",
    "sampler_factory",
    "collate_factory",
]


def reader_factory(reader_cfg):
    """Instantiates reader based on type specified in configuration under
    `io.reader.name`. The name must match the name of a class under
    `spine.io.readers`.

    Parameters
    ----------
    reader_cfg : dict
        Writer configuration dictionary

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    # Initialize reader
    return instantiate(READER_DICT, reader_cfg)


def writer_factory(writer_cfg, prefix=None, split=False):
    """Instantiates writer based on type specified in configuration under
    `io.writer.name`. The name must match the name of a class under
    `spine.io.writers`.

    Parameters
    ----------
    writer_cfg : dict
        Writer configuration dictionary
    prefix : str, optional
        Input file prefix to use as an output name
    split : bool, default False
        Split the output into one file per input file

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    # Initialize writer
    return instantiate(WRITER_DICT, writer_cfg, prefix=prefix, split=split)


def loader_factory(
    dataset,
    dtype,
    geo=None,
    batch_size=None,
    minibatch_size=None,
    shuffle=True,
    sampler=None,
    num_workers=0,
    collate_fn=None,
    entry_list=None,
    distributed=False,
    world_size=0,
    rank=0,
    **kwargs,
):
    """Instantiates a PyTorch DataLoader based on configuration."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to use loader_factory.")

    from torch.utils.data import DataLoader

    # Process the batch size, make sure it is sensible
    assert (batch_size is not None) ^ (
        minibatch_size is not None
    ), "Provide either `batch_size` or `minibatch_size`, not both."

    if batch_size is not None:
        assert (
            world_size == 0 or (batch_size % world_size) == 0
        ), "The batch_size must be a multiple of the number of GPUs."
        minibatch_size = batch_size // max(world_size, 1)
    elif minibatch_size is not None:
        batch_size = minibatch_size * max(world_size, 1)

    # Initialize the dataset
    torch_dataset = dataset_factory(dataset, entry_list, dtype, geo=geo)

    # Initialize the sampler
    if sampler is not None:
        sampler = sampler_factory(
            sampler, torch_dataset, batch_size, distributed, world_size, rank
        )

    # Initialize the collate function
    if collate_fn is not None:
        collate_fn = collate_factory(
            collate_fn, torch_dataset.data_types, torch_dataset.overlay_methods
        )

    # Initialize the loader
    return DataLoader(
        torch_dataset,
        batch_size=minibatch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def dataset_factory(dataset_cfg, entry_list=None, dtype=None, geo=None):
    """Instantiates a Dataset based on a configuration."""
    from . import dataset

    dataset_dict = module_dict(dataset)

    # Append the entry_list if it is provided independently
    if entry_list is not None:
        warn(
            "You are manually overwriting the existing `entry_list` "
            "argument provided in the configuration file."
        )
        dataset_cfg["entry_list"] = entry_list

    # Initialize dataset
    return instantiate(dataset_dict, dataset_cfg, dtype=dtype, geo=geo)


def sampler_factory(
    sampler_cfg, dataset, minibatch_size, distributed=False, num_replicas=1, rank=0
):
    """Instantiates sampler based on configuration."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to use sampler_factory.")

    from . import sample

    sampler_dict = module_dict(sample)

    # Initialize sampler
    sampler_obj = instantiate(
        sampler_dict, sampler_cfg, dataset=dataset, batch_size=minibatch_size
    )

    # If we are working a distributed environment, wrap the sampler
    if distributed:
        sampler_obj = sample.DistributedProxySampler(sampler_obj, num_replicas, rank)

    return sampler_obj


def collate_factory(collate_cfg, data_types, overlay_methods):
    """Instantiates collate function based on configuration."""
    from . import collate

    collate_dict = module_dict(collate)

    return instantiate(
        collate_dict,
        collate_cfg,
        "collate_fn",
        data_types=data_types,
        overlay_methods=overlay_methods,
    )
