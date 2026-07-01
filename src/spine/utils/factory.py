"""Contains functions needed to instantiate classes from configuration blocks.

This allows to generically convert a YAML block into an instatiated class
with all the appropriate checks that the class exists and is provided
with appropriate arguments.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from types import ModuleType
from typing import Any, TypeAlias
from warnings import warn

from .logger import logger

Registry: TypeAlias = dict[str, Any]
Config: TypeAlias = Mapping[str, Any] | str
ModuleSpec: TypeAlias = dict[str, Any]
ParsedModules: TypeAlias = OrderedDict[str, ModuleSpec]


def module_dict(
    module: ModuleType, class_name: str | None = None, pattern: str | None = None
) -> Registry:
    """Converts module into a dictionary which maps class names onto classes.

    Parameters
    ----------
    module : module
        Module from which to fetch the classes
    class_name : str, optional
        If specified, only allow aliases that match it
    pattern : str, optional
        If specified, looks for a specific pattern in the class name

    Returns
    -------
    dict
        Dictionary which maps acceptable class names to classes themselves
    """
    # Loop over classes/functions in the module
    mod_dict: Registry = {}
    cls_names = getattr(module, "__attr__", dir(module))
    for cls_name in cls_names:
        # Skip private objects
        if cls_name[0] == "_":
            continue

        # If a pattern is specified, check for it in the class name
        cls = getattr(module, cls_name)
        if pattern is not None and pattern not in cls.__name__:
            continue

        # Only consider classes which belong to the module of interest
        if hasattr(cls, "__module__") and module.__name__ in cls.__module__:

            # Store the class name as an option to fetch it
            mod_dict[cls_name] = cls

            # If a name is provided, add it to the allowed options
            name = getattr(cls, "name", None)
            if name:
                mod_dict[name] = cls

            # If aliases are specified, it is allowed but should be avoided
            if hasattr(cls, "aliases"):
                for al in cls.aliases:
                    if class_name is not None and class_name == al:
                        warn(
                            f"This name ({al}) is deprecated. Use "
                            f"{cls.name} instead.",
                            DeprecationWarning,
                        )
                        mod_dict[al] = cls
                    else:
                        mod_dict[al] = cls

    return mod_dict


def instantiate(
    mod_dict: Registry, cfg: Config, alt_name: str | None = None, **kwargs: Any
) -> Any:
    """Instantiates a class based on a configuration dictionary and a list of
    possible classes to chose from.

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
    mod_dict : dict
        Dictionary which maps a class name onto an object class.
    cfg : dict
        Configuration dictionary
    alt_name : str, optional
        Key under which the class name can be specfied, beside 'name' itself
    **kwargs : dict, optional
        Additional parameters to pass to the function

    Returns
    -------
    object
        Instantiated object
    """
    # If the configuration is a string, assume it is a class name with no
    # parameters to be passed to it
    if isinstance(cfg, str):
        config: dict[str, Any] = {"name": cfg}
    else:
        config = dict(deepcopy(cfg))

    # Get the name of the class, check that it exists
    if alt_name is not None:
        if (alt_name in config) == ("name" in config):
            raise ValueError(f"Should specify one of `name` or `{alt_name}`")
        name = alt_name if alt_name in config else "name"
    else:
        if "name" not in config:
            raise ValueError("Could not find the name of the class under `name`")
        name = "name"

    class_name = config.pop(name)

    # Check that the class we are looking for exists
    if class_name not in mod_dict:
        valid_keys = list(mod_dict.keys())
        raise ValueError(
            f"Could not find '{class_name}' in the dictionary "
            f"which maps names to classes. Available names: "
            f"{valid_keys}"
        )

    # Gather the arguments and keyword arguments to pass to the function
    args = config.pop("args", [])
    kwargs = dict(config.pop("kwargs", {}), **kwargs)

    # If args is specified as a dictionary, append it to kwargs (deprecated)
    if isinstance(args, dict):
        warn(
            "If specifying keyword arguments, should use `kwargs` instead "
            f"of `args` in {class_name}",
            category=DeprecationWarning,
        )
        for key in args.keys():
            if key in kwargs:
                raise ValueError(
                    f"The keyword argument {key} is provided under "
                    "`args` and `kwargs`. Ambiguous."
                )
        kwargs.update(args)
        args = []

    # If some arguments were specified at the top level, append them
    for key in config:
        if key in kwargs:
            raise ValueError(
                f"The keyword argument {key} is provided "
                "at the top level and under `kwargs`. Ambiguous."
            )
    kwargs.update(config)

    # Intialize
    cls = mod_dict[class_name]
    try:
        return cls(*args, **kwargs)

    except Exception as err:
        logger.error(
            "Failed to instantiate %s with these arguments:\n"
            "  - args: %s\n  - kwargs: %s",
            cls.__name__,
            args,
            kwargs,
        )

        raise err


def parse_module_config(
    modules: Mapping[str, Mapping[str, Any] | None],
    name_key: str = "name",
    priority_key: str = "priority",
    sort_by_priority: bool = False,
    priority_descending: bool = False,
    skip_none: bool = True,
) -> ParsedModules:
    """Parse an ordered mapping of module blocks.

    Each top-level key is treated as the module label. The concrete class name
    is read from ``name_key`` when present, otherwise the label itself is used
    as the class name. This supports both compact blocks such as:

    .. code-block:: yaml

        gain:
          gain: 2.0

    and repeated instances of the same module:

    .. code-block:: yaml

        first_gain:
          name: gain
          gain: 2.0
        second_gain:
          name: gain
          gain: 3.0

    Parameters
    ----------
    modules : Mapping
        Ordered mapping of module labels to configuration dictionaries.
    name_key : str, default 'name'
        Configuration key which specifies the class name.
    priority_key : str, default 'priority'
        Configuration key which specifies optional execution priority.
    sort_by_priority : bool, default False
        If ``True``, modules with smaller priority values run first. Modules
        without a priority retain their relative order after prioritized
        modules.
    priority_descending : bool, default False
        If ``True`` and ``sort_by_priority`` is ``True``, modules with larger
        priority values run first instead.
    skip_none : bool, default True
        If ``True``, skip entries explicitly set to ``None``.

    Returns
    -------
    OrderedDict
        Mapping of module label to dictionaries with ``name``, ``cfg`` and
        ``priority`` fields.
    """
    if not isinstance(modules, Mapping):
        raise TypeError("Module configuration must be a mapping.")

    parsed: list[tuple[int, str, str, int | float | None, dict[str, Any]]] = []
    for index, (label, cfg) in enumerate(modules.items()):
        if cfg is None and skip_none:
            continue
        if not isinstance(cfg, Mapping):
            raise TypeError(f"Configuration for module `{label}` must be a mapping.")

        config = deepcopy(dict(cfg))
        name = config.pop(name_key, label)
        priority = config.pop(priority_key, None)
        parsed.append((index, label, name, priority, config))

    if sort_by_priority:
        if priority_descending:
            parsed.sort(
                key=lambda item: (
                    item[3] is None,
                    -item[3] if item[3] is not None else item[0],
                    item[0],
                )
            )
        else:
            parsed.sort(
                key=lambda item: (
                    item[3] is None,
                    item[3] if item[3] is not None else item[0],
                    item[0],
                )
            )

    return OrderedDict(
        (label, {"name": name, "cfg": config, "priority": priority})
        for _, label, name, priority, config in parsed
    )


def instantiate_modules(
    mod_dict: Registry,
    modules: Mapping[str, Mapping[str, Any] | None],
    name_key: str = "name",
    priority_key: str = "priority",
    sort_by_priority: bool = False,
    priority_descending: bool = False,
    skip_none: bool = True,
    **kwargs: Any,
) -> OrderedDict[str, Any]:
    """Instantiate an ordered mapping of module configuration blocks.

    Parameters
    ----------
    mod_dict : dict
        Dictionary which maps class names onto classes.
    modules : Mapping
        Ordered mapping of module labels to configuration dictionaries.
    name_key : str, default 'name'
        Configuration key which specifies the class name.
    priority_key : str, default 'priority'
        Configuration key which specifies optional execution priority.
    sort_by_priority : bool, default False
        If ``True``, modules with smaller priority values run first.
    priority_descending : bool, default False
        If ``True`` and ``sort_by_priority`` is ``True``, modules with larger
        priority values run first instead.
    skip_none : bool, default True
        If ``True``, skip entries explicitly set to ``None``.
    **kwargs : dict
        Extra keyword arguments forwarded to every instantiated class.

    Returns
    -------
    OrderedDict
        Mapping of module label to instantiated module objects.
    """
    parsed = parse_module_config(
        modules,
        name_key=name_key,
        priority_key=priority_key,
        sort_by_priority=sort_by_priority,
        priority_descending=priority_descending,
        skip_none=skip_none,
    )

    instances: OrderedDict[str, Any] = OrderedDict()
    for label, spec in parsed.items():
        cfg = dict(spec["cfg"])
        cfg[name_key] = spec["name"]
        instances[label] = instantiate(mod_dict, cfg, **kwargs)

    return instances
