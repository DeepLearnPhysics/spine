from copy import deepcopy
from functools import partial
from warnings import warn

from torch.utils.data import DataLoader


def loader_factory(dataset, minibatch_size, shuffle=True,
        sampler=None, num_workers=0, collate_fn=None, collate=None,
        event_list=None, distributed=False, world_size=1, rank=0):
    '''
    Instantiates a DataLoader based on configuration.

    Dataset comes from `dataset_factory`.

    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary
    minibatch_size : int
        Number of data samples to load per iteration, per process
    num_workers : bool, default 0
        Number of CPU cores to use to load data. If 0, the process which
        runs the model will also load the data.
    shuffle : bool, default True
        If True, shuffle the dataset entries
    sampler : str, optional
        Name of the function used to sample data into batches
    collate_fn : str, optional
        Name of the function used to collate data into batches
    collate : dict, optional
        Dictionary of collate function and collate parameters, if any
    event_list : list, optional
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
    '''
    # Initialize the dataset
    dataset = dataset_factory(dataset, event_list)

    # Initialize the sampler
    if sampler is not None:
        sampler['minibatch_size'] = minibatch_size
        sampler = sampler_factory(sampler, dataset,
                distributed, world_size, rank)

    # Initialize the collate function
    assert not (collate_fn is not None and collate is not None), \
            'Must specify either `collate_fn` or `collate`, not both'
    if collate_fn is not None:
        collate = {'name': collate_fn}
    if collate is not None:
        collate_fn = collate_factory(collate)

    # Initialize the loader
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=shuffle,
            sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)

    return loader


def dataset_factory(dataset_cfg, event_list=None):
    '''
    Instantiates dataset based on type specified in configuration under
    `iotool.dataset.name`. The name must match the name of a class under
    `mlreco.iotools.datasets`.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration dictionary
    event_list: list, optional
        List of entry numbers to include in the dataset

    Returns
    -------
    torch.utils.data.Dataset
        Initialized dataset

    Note
    ----
    Currently the choice is limited to `LArCVDataset` only.
    '''
    # If there is no name in the dataset configuration, cannot instantiate
    if 'name' not in dataset_cfg:
        raise KeyError('Must specify a dataset name under the dataset block')

    # Fetch dataset keyword arguments
    kwargs = deepcopy(dataset_cfg)
    name   = kwargs.pop('name')
    if event_list is not None:
        kwargs['event_list'] = event_list
    
    # Initialize dataset
    import mlreco.iotools.datasets
    dataset = getattr(mlreco.iotools.datasets, name)(**kwargs)

    return dataset


def sampler_factory(sampler_cfg, dataset, distributed = False, num_replicas = 1, rank = 0):
    '''
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
    '''
    # If there is no name in the sampler configuration, cannot instantiate
    if 'name' not in sampler_cfg:
        raise KeyError('Must specify a sampler name under the sampler block')

    # Fetch sampler keyword arguments
    kwargs = deepcopy(sampler_cfg)
    name   = kwargs.pop('name')

    # Add the dataset to the arguments
    kwargs['dataset'] = dataset

    # If distributed, provide additional arguments
    if distributed:
        kwargs['num_replicas'] = num_replicas
        kwargs['rank'] = rank
    else:
        kwargs['data_source'] = None # Vestigial, will break with pytorch 2.2

    # Initialize sampler
    import mlreco.iotools.samplers
    sampler = getattr(mlreco.iotools.samplers, name)(distributed)(**kwargs)

    return sampler


def collate_factory(collate_cfg):
    '''
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
    '''
    # If there is no name in the collate configuration, cannot instantiate
    # Currently also handles deprecated nomenclature
    if 'collate_fn' in collate_cfg:
        msg = ('Specify the collate function name under the `name` '
               'key of the `collate` block, not the `collate_fn` key')
        warn(msg, DeprecationWarning, stacklevel=2)
        collate_cfg['name'] = collate_cfg.pop('collate_fn')
    if 'name' not in collate_cfg:
        raise KeyError('Must specify a collate name under the collate block')

    # Fetch collate function keyword arguments
    kwargs = deepcopy(collate_cfg)
    name   = kwargs.pop('name')

    # Initialize collate function
    import mlreco.iotools.collates
    collate = partial(getattr(mlreco.iotools.collates, name), **kwargs)

    return collate


def writer_factory(name, **writer_args):
    '''
    Instantiates writer based on type specified in configuration under
    `iotool.writer.name`. The name must match the name of a class under
    `mlreco.iotools.writers`.

    Parameters
    ----------
    name : str
        Name of the writer class
    **writer_args : dict
        Writer configuration dictionary

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    '''
    # Initialize writer
    import mlreco.iotools.writers
    writer = getattr(mlreco.iotools.writers, name)(**writer_args)

    return writer
