"""Contains functions needed to instantiate a class from a dictionary.

This allows to generically convert a YAML block into an instatiated class
with all the appropriate checks that the class exists and is provided
with appropriate arguments.
"""

from copy import deepcopy
from warnings import warn


def module_dict(module, class_name=None):
    """Converts module into a dictionary which maps class names onto classes.

    Parameters
    ----------
    module : module
        Module from which to fetch the classes
    class_name : str, optional
        If specified, only allow aliases that match it

    Returns
    -------
    dict
        Dictionary which maps acceptable class names to classes themselves
    """
    # Loop over classes/functions in the module
    module_dict = {}
    for cls_name in dir(module):
        # Skip private objects
        if cls_name[0] == '_':
            continue

        # Only consider classes which belong to the module of intest
        cls = getattr(module, cls_name)
        if (hasattr(cls, '__module__') and
            module.__name__ in cls.__module__):
            # Store the class name as an option to fetch it
            module_dict[cls_name] = cls

            # If a name is provided, add it to the allowed options
            if hasattr(cls, 'name') and len(cls.name):
                module_dict[cls.name] = cls

            # If aliases are specified, it is allowed but should be avoided
            if hasattr(cls, 'aliases'):
                for al in cls.aliases:
                    if class_name is not None and class_name == al:
                        warn(f"This name ({al}) is deprecated. Use "
                             f"{cls.name} instead.", DeprecationWarning,
                             stacklevel=1)
                        module_dict[al] = cls
                    else:
                        module_dict[al] = cls

    return module_dict


def instantiate(module_dict, cfg, name=None):
    """Instantiates an instance of a class based on a configuration dictionary
    and a list of possible classes to chose from.
    
    This function supports two YAML configuration structures
    (parsed as a dictionary):
    
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
            value_1
            value_2
            ...
          kwargs:
            kwarg_1: value_1
            kwarg_2: value_2
            ...

    The `name` field can have a different name, as long as it is specified.

    Parameters
    ----------
    module_dict : dict
        Dictionary which maps a class name onto an object class.
    cfg : dict
        Configuration dictionary
    name : str, optional
        Key under which the class name can be specfied

    Returns
    -------
    object
        Instantiated object
    """
    # Get the name of the class, check that it exists
    config = deepcopy(cfg)
    if name is not None:
        assert (name in config) ^ ('name' in config), (
                f"Should specify one of `name` or `{name}`")
        name = name if name in config else 'name'
    else:
        assert 'name' in config, (
                "Could not find the name of the class under `name`")
        name = 'name'
    
    class_name = config.pop(name)

    # Check that the class we are looking for exists
    if class_name not in module_dict:
        valid_keys = list(module_dict.keys())
        raise ValueError(f"Could not find {class_name} in the dictionary "
                         f"which maps names to classes. Available names: "
                         f"{valid_keys}")

    # Gather the arguments and keyword arguments to pass to the function
    args = config.pop('args', [])
    kwargs = config.pop('kwargs', {})

    # If args is specified as a dictionary, append it to kwargs (deprecated)
    if isinstance(args, dict):
        # TODO: proper logging
        warn("If specifying keyword arguments, should use `kwargs` instead "
             "of args in {class_name}", category=DeprecationWarning,
             stacklevel=1)
        for key in args.keys():
            assert key not in kwargs, (
                    f"The keyword argument {key} is provided under "
                     "`args` and `kwargs`. Ambiguous.")
        kwargs.update(args)
        args = []

    # If some arguments were specified at the top level, append them
    for key in config.keys():
        assert key not in kwargs, (
                f"The keyword argument {key} is provided "
                 "at the top level and under `kwargs`. Ambiguous.")
    kwargs.update(config)

    # Intialize
    try:
        return module_dict[class_name](*args, **kwargs)

    except Exception as err:
        # TODO: proper logging
        print(f"Failed to instantiate {class_name} with these arguments:\n"
              f"  - args: {args}\n  - kwargs: {kwargs}")
        raise err
