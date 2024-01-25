import numpy as np
import torch

from copy import deepcopy
from warnings import warn


def cycle(data_io):
    '''
    Use this function instead of itertools.cycle to avoid creating a memory
    leak (itertools.cycle attempts to save all outputs in order to re-cycle
    through them)

    Parameters
    ----------
    data_io : torch.utils.data.DataLoader
        Data loader
    '''
    while True:
        for x in data_io:
            yield x


def to_numpy(array):
    '''
    Function which casts an array-like object
    to a `numpy.ndarray`.

    Parameters
    ----------
    array : object
        Array-like object (can be either `np.ndarray`, `torch.Tensor`
        or `ME.SparseTensor`)

    Returns
    -------
    np.ndarray
        Array cast to np.ndarray
    '''
    import MinkowskiEngine as ME

    if isinstance(array, (list, tuple)):
        return np.array(array)
    elif isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        if array.ndim == 0:
            return array.item()
        else:
            return array.cpu().detach().numpy()
    elif isinstance(array, ME.SparseTensor):
        return torch.cat([array.C.float(), array.F], dim=1).detach().cpu().numpy()
    else:
        raise TypeError('Unknown return type %s' % type(array))


def local_cdist(v1, v2):
    '''
    Function which computes the pairwise distances between two
    collections of points stored as `torch.Tensor` objects.

    This is necessary because the torch.cdist implementation is either
    slower (with the `donot_use_mm_for_euclid_dist` option) or produces
    dramatically wrong answers under certain situations (with the
    `use_mm_for_euclid_dist_if_necessary` option).

    Parameters
    ----------
    v1 : torch.Tensor
        (N, D) tensor of coordinates
    v2 : torch.Tensor
        (M, D) tensor of coordinates

    Returns
    -------
    torch.Tensor
        (N, M) tensor of pairwise distances
    '''
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def instantiate(module, cfg, name = 'name'):
    '''
    Instantiates an instance of a class based on a configuration dictionary
    and a list of possible classes. This function supports two YAML
    configuration structures (parsed as a dictionary):
    
    .. code-block:: yaml

        function:
          name: function_name
          kwarg_1: value_1
          kwarg_2: value_2
          ...

    or

    .. code-block:: yaml

        function:
          name: function_name
          args:
            kwarg_1: value_1
            kwarg_2: value_2
            ...

    The `name` field can have a different name, as long as it is specified.

    Parameters
    ----------
    module : Union[module, dict]
        Module from which to fetch the classes or dictionary which maps
        a function name onto an object class.
    cfg : dict
        Configuration dictionary

    Returns
    -------
    object
        Instantiated object
    '''
    # Get the name of the class, check that it exists
    config = deepcopy(cfg)
    try:
        class_name = config.pop(name)
    except Exception as err:
        # TODO: proper logging
        print('Could not find the name of the function ' \
                f'to initialize under {name}')
        raise err

    if isinstance(module, dict) and class_name not in module:
        valid_keys = list(module.keys())
        raise ValueError(f'Could not find {class_name} in the dictionary ' \
                f'which maps names to classes. Available names: {valid_keys}')
    elif not isinstance(module, dict) and not hasattr(module, class_name):
        raise ValueError(f'Could not find {class_name} in the provided module')

    # Gather the arguments and keyword arguments to pass to the function
    args = config.pop('args', [])
    kwargs = config.pop('kwargs', {})

    # If args is specified as a dictionary, append it to kwargs (deprecated)
    if isinstance(args, dict):
        # TODO: proper logging
        warn('If specifying keyword arguments, should use `kwargs` instead '\
                'of args in {class_name}', category = DeprecationWarning)
        for key in args.keys():
            assert key not in kwargs, f'The keyword argument {key} is ' \
                    'provided under `args` and `kwargs`. Ambiguous.'
        kwargs.update(args)
        args = []

    # If some arguments were specified at the top level, append them
    for key in config.keys():
        assert key not in kwargs, f'The keyword argument {key} is provided ' \
                'at the top level and under `kwargs`. Ambiguous.'
    kwargs.update(config)

    # Intialize
    try:
        if isinstance(module, dict):
            return module[class_name](*args, **kwargs)
        else:
            return getattr(module, class_name)(*args, **kwargs)
    except Exception as err:
        # TODO: proper logging
        print(f'Failed to instantiate {class_name} with these arguments:\n' \
                f'  - args: {args}\n  - kwargs: {kwargs}')
        raise err
