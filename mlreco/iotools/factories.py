"""Functions that instantiate IO tools classes from configuration blocks."""

from copy import deepcopy
from warnings import warn
from inspect import isfunction

from torch.utils.data import DataLoader

from mlreco.utils.factory import module_dict, instantiate

from . import datasets, samplers, collates, readers, writers

DATASET_DICT = module_dict(datasets)
COLLATE_DICT = module_dict(collates)
READER_DICT  = module_dict(readers)
WRITER_DICT  = module_dict(writers)


def loader_factory(dataset, batch_size, shuffle=True,
                   sampler=None, num_workers=0, collate_fn=None,
                   entry_list=None, distributed=False, world_size=1, rank=0):
    """Instantiates a DataLoader based on configuration.

    Dataset comes from `dataset_factory`.

    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary
    batch_size : int
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
        Total number of processes running the sampler
    rank : int, default 0
        Unique identifier of the process sampling data

    Returns
    -------
    torch.utils.data.DataLoader
        Initialized dataloader
    """
    # Initialize the dataset
    dataset = dataset_factory(dataset, entry_list)

    # Initialize the sampler
    if sampler is not None:
        sampler['batch_size'] = batch_size
        sampler = sampler_factory(sampler, dataset, distributed,
                                  world_size, rank)

    # Initialize the collate function
    if collate_fn is not None:
        collate_fn = collate_factory(collate_fn)

    # Initialize the loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        sampler=sampler, num_workers=num_workers,
                        collate_fn=collate_fn)

    return loader


def dataset_factory(dataset_cfg, entry_list=None):
    """Instantiates a Dataset based on a configuration.
    
    The Dataset type is specified in configuration under `iotool.dataset.name`.
    The name must match the name of a class under `mlreco.iotools.datasets`.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration dictionary
    entry_list: list, optional
        List of entry numbers to include in the dataset

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
        # TODO: if it already exists: issue warning, log properly
        dataset_cfg['entry_list'] = entry_list

    # Tolerate some deprecated arguments, for now
    # TODO: remove this possibility
    if 'data_keys' in dataset_cfg:
        warn("Use `file_keys` instead of `data_keys`", DeprecationWarning)
        dataset_cfg['file_keys'] = dataset_cfg.pop('data_keys')
    if 'event_list' in dataset_cfg:
        warn("Use `entry_list` instead of `event_list`", DeprecationWarning)
        dataset_cfg['entry_list'] = dataset_cfg.pop('event_list')
    
    # Initialize dataset
    return instantiate(DATASET_DICT, dataset_cfg)


def sampler_factory(sampler_cfg, dataset, distributed=False,
                    num_replicas=1, rank=0):
    """
    Instantiates sampler based on type specified in configuration under
    `iotool.sampler.name`. The name must match the name of a class under
    `mlreco.iotools.samplers`.

    Parameters
    ----------
    sampler_cfg : dict
        Sampler configuration dictionary
    dataset : torch.utils.data.Dataset
        Dataset to sample from
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
    # Get the list of available samplers (must inherit from the right class)
    sampler_dict = {}
    for sampler in dir(samplers):
        if 'Sampler' in sampler:
            cls = getattr(samplers, sampler)
            if isfunction(cls):
                sampler_dict[sampler] = cls(distributed)
                if hasattr(sampler_dict[sampler], 'name'):
                    name = sampler_dict[sampler].name
                    sampler_dict[name] = sampler_dict[sampler]
            else:
                sampler_dict[sampler] = cls

    # Fetch sampler keyword arguments
    config = deepcopy(sampler_cfg)

    # Add the dataset to the arguments
    config['dataset'] = dataset

    # If distributed, provide additional arguments
    if distributed:
        config['num_replicas'] = num_replicas
        config['rank'] = rank
    else:
        config['data_source'] = None # Vestigial, will break with pytorch 2.2

    # Initialize sampler
    return instantiate(sampler_dict, config)


def collate_factory(collate_cfg):
    """
    Instantiates collate function based on type specified in configuration
    under `iotool.collate.name`. The name must match the name of a class
    under `mlreco.iotools.collates`.

    Parameters
    ----------
    collate_cfg : dict
        Collate function configuration dictionary

    Returns
    -------
    function
        Initialized collate function
    """
    # Initialize collate function
    return instantiate(COLLATE_DICT, collate_cfg, 'collate_fn')


def reader_factory(reader_cfg):
    """
    Instantiates reader based on type specified in configuration under
    `iotool.reader.name`. The name must match the name of a class under
    `mlreco.iotools.readers`.

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


def writer_factory(writer_cfg):
    """
    Instantiates writer based on type specified in configuration under
    `iotool.writer.name`. The name must match the name of a class under
    `mlreco.iotools.writers`.

    Parameters
    ----------
    writer_cfg : dict
        Writer configuration dictionary

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    # Initialize writer
    return instantiate(WRITER_DICT, writer_cfg)
