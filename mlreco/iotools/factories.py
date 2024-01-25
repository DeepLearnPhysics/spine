from copy import deepcopy
from warnings import warn

from torch.utils.data import DataLoader

from mlreco.utils import instantiate


def loader_factory(dataset, batch_size, shuffle=True,
        sampler=None, num_workers=0, collate_fn=None, collate=None,
        event_list=None, distributed=False, world_size=1, rank=0):
    '''
    Instantiates a DataLoader based on configuration.

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
        sampler['batch_size'] = batch_size
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)

    return loader


def dataset_factory(dataset_cfg, event_list = None):
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
    # Append the event_list if it is provided independently
    if event_list is not None:
        # TODO: if it already exists: issue warning, log properly
        dataset_cfg['event_list'] = event_list
    
    # Initialize dataset
    from mlreco.iotools import datasets
    return instantiate(datasets, dataset_cfg)


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
        # TODO: log properly
        msg = ('Specify the collate function name under the `name` '
               'key of the `collate` block, not the `collate_fn` key')
        warn(msg, DeprecationWarning, stacklevel=2)
        collate_cfg['name'] = collate_cfg.pop('collate_fn')
    if 'name' not in collate_cfg:
        raise KeyError('Must specify a collate name under the collate block')

    # Initialize collate function
    from mlreco.iotools import collates

    return instantiate(collates, collate_cfg)


def writer_factory(writer_cfg):
    '''
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
    '''
    # Initialize writer
    from mlreco.iotools import writers

    return instantiate(writers, writer_cfg)
