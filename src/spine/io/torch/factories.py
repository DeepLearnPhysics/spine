"""Functions that instantiate PyTorch-dependent IO tools from configuration blocks."""

from warnings import warn

from torch.utils.data import DataLoader

from spine.utils.factory import instantiate, module_dict

from . import collate, dataset, sample

DATASET_DICT = module_dict(dataset)
SAMPLER_DICT = module_dict(sample)
COLLATE_DICT = module_dict(collate)

__all__ = ["loader_factory", "dataset_factory", "sampler_factory", "collate_factory"]


def loader_factory(
    dataset,
    dtype,
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
):
    """Instantiates a DataLoader based on configuration.

    Dataset comes from `dataset_factory`.

    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary
    dtype : str
        Data type to cast the input data to
    batch_size : int, optional
        Number of data samples to load per iteration
    minibatch_size : int, optional
        Number of data samples to load per iteration, per process
    num_workers : bool, default 0
        Number of CPU cores to use to load data. If 0, the process which
        runs the model will also load the data.
    shuffle : bool, default True
        If True, shuffle the dataset entries
    sampler : str, optional
        Name of the function used to sample data into batches
    collate_fn : dict, optional
        Dictionary of collate function and collate parameters, if any
    entry_list : list, optional
        List of entry numbers to include in the dataset
    distributed : bool, default False
        If True, the loader will be prepared for distributed execution
    world_size : int, default 1
        Total number of GPUs using the sampler
    rank : int, default 0
        Unique identifier of the process sampling data

    Returns
    -------
    torch.utils.data.DataLoader
        Initialized dataloader
    """
    # Process the batch size, make sure it is sensible
    assert (batch_size is not None) ^ (
        minibatch_size is not None
    ), "Provide either `batch_size` or `minibatch_size`, not both."

    if batch_size is not None:
        assert (
            world_size == 0 or (batch_size % world_size) == 0
        ), "The batch_size must be a multiple of the number of GPUs."
        minibatch_size = batch_size // max(world_size, 1)

    # Initialize the dataset
    dataset = dataset_factory(dataset, entry_list, dtype)

    # Initialize the sampler
    if sampler is not None:
        sampler = sampler_factory(
            sampler, dataset, batch_size, distributed, world_size, rank
        )

    # Initialize the collate function
    if collate_fn is not None:
        collate_fn = collate_factory(
            collate_fn, dataset.data_types, dataset.overlay_methods
        )

    # Initialize the loader
    loader = DataLoader(
        dataset,
        batch_size=minibatch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return loader


def dataset_factory(dataset_cfg, entry_list=None, dtype=None):
    """Instantiates a Dataset based on a configuration.

    The Dataset type is specified in configuration under `io.dataset.name`.
    The name must match the name of a class under `spine.io.datasets`.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration dictionary
    entry_list: list, optional
        List of entry numbers to include in the dataset
    dtype : str, optional
        Data type to cast the input data to (to match the downstream model)

    Returns
    -------
    torch.utils.data.Dataset
        Initialized dataset

    Note
    ----
    Currently the choice is limited to `LArCVDataset` only.
    """
    # Append the entry_list if it is provided independently
    if entry_list is not None:
        warn(
            "You are manually overwriting the existing `entry_list` "
            "argument provided in the configuration file."
        )
        dataset_cfg["entry_list"] = entry_list

    # Initialize dataset
    return instantiate(DATASET_DICT, dataset_cfg, dtype=dtype)


def sampler_factory(
    sampler_cfg, dataset, minibatch_size, distributed=False, num_replicas=1, rank=0
):
    """Instantiates sampler based on type specified in configuration under
    `io.sampler.name`. The name must match the name of a class under
    `spine.io.sample`.

    Parameters
    ----------
    sampler_cfg : dict
        Sampler configuration dictionary
    dataset : torch.utils.data.Dataset
        Dataset to sample from
    minibatch_size : int
        Number of data samples to load per iteration, per process
    distributed: bool, default False
        If True, initialize as a DistributedSampler
    num_replicas : int, default 1
        Total number of processes running the sampler
    rank : int, default 0
        Unique identifier of the process sampling data

    Returns
    -------
    Union[torch.utils.data.Sampler, torch.utils.data.DistributedSampler]
        Initialized sampler
    """
    # Initialize sampler
    sampler = instantiate(
        SAMPLER_DICT, sampler_cfg, dataset=dataset, batch_size=minibatch_size
    )

    # If we are working a distributed environment, wrap the sampler
    if distributed:
        sampler = sample.DistributedProxySampler(sampler, num_replicas, rank)

    # Return
    return sampler


def collate_factory(collate_cfg, data_types, overlay_methods):
    """Instantiates collate function based on type specified in configuration
    under `io.collate.name`. The name must match the name of a class
    under `spine.io.collates`.

    Parameters
    ----------
    collate_cfg : dict
        Collate function configuration dictionary
    data_types : dict
        Dictionary of data types
    overlay_methods : dict
        Dictionary of overlay methods

    Returns
    -------
    function
        Initialized collate function
    """
    # Initialize collate function
    return instantiate(
        COLLATE_DICT,
        collate_cfg,
        "collate_fn",
        data_types=data_types,
        overlay_methods=overlay_methods,
    )
