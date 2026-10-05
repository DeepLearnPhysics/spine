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

from spine.logging import logger

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


def resolve_module_provider(
    config: Mapping[str, Any] | str,
    *,
    default: str | None = None,
    aliases: tuple[str, ...] = ("name",),
    warn_deprecated: bool = True,
) -> str | None:
    """Resolve an implementation selector without mutating configuration.

    Parameters
    ----------
    config : Mapping or str
        Component descriptor or short provider name.
    default : str, optional
        Provider inferred from a module mapping key when no selector is given.
    aliases : tuple of str
        Deprecated selector spellings accepted in this context.
    warn_deprecated : bool, default True
        Emit a warning for legacy selectors. Routing code can disable repeated
        warnings before the actual construction boundary validates the config.

    Returns
    -------
    str or None
        Selected implementation, or the default if no selector is present.

    Raises
    ------
    ValueError
        If selectors conflict or an explicit selector is not a nonempty string.
    """
    if isinstance(config, str):
        value = config
    else:
        keys = set(("provider", *aliases)).intersection(config)
        if len(keys) > 1:
            raise ValueError(
                f"Specify only one of the implementation selectors: {sorted(keys)}."
            )
        if not keys:
            return default
        key = next(iter(keys))
        value = config[key]
        if key != "provider" and warn_deprecated:
            warn(
                f"Implementation selector `{key}` is deprecated; use `provider` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Implementation `provider` must be a nonempty string.")
    return value


def instantiate(
    mod_dict: Registry, cfg: Config, alt_name: str | None = None, **kwargs: Any
) -> Any:
    """Instantiates a class based on a configuration dictionary and a list of
    possible classes to chose from.

    This function supports two YAML configuration structures
    (parsed as a dictionary):

    .. code-block:: yaml

        function:
          provider: function_name
          kwarg_1: value_1
          kwarg_2: value_2
          ...

    or

    .. code-block:: yaml

        function:
          provider: function_name
          config:
            kwarg_1: value_1
            kwarg_2: value_2
            ...

    The canonical selector is `provider`. Legacy `name` and the optional
    context-specific alias remain accepted with deprecation warnings.

    Parameters
    ----------
    mod_dict : dict
        Dictionary which maps a class name onto an object class.
    cfg : dict
        Configuration dictionary
    alt_name : str, optional
        Context-specific deprecated implementation selector alias.
    **kwargs : dict, optional
        Additional parameters to pass to the function

    Returns
    -------
    object
        Instantiated object
    """
    # Resolve deprecated spellings at the public construction boundary.
    aliases = ("name",) if alt_name is None else ("name", alt_name)
    class_name = resolve_module_provider(cfg, aliases=aliases)
    if class_name is None:
        raise ValueError(
            "Component configuration requires `provider` (legacy `name` is also accepted)."
        )
    config = {} if isinstance(cfg, str) else deepcopy(dict(cfg))
    for selector in ("provider", *aliases):
        config.pop(selector, None)

    # Check that the class we are looking for exists
    if class_name not in mod_dict:
        valid_keys = list(mod_dict.keys())
        raise ValueError(
            f"Could not find '{class_name}' in the dictionary "
            f"which maps names to classes. Available names: "
            f"{valid_keys}"
        )

    # YAML parameters and runtime-injected dependencies share no precedence:
    # a duplicate is an error regardless of where it was supplied.
    parameters = extract_module_parameters(config, context=f"Module `{class_name}`")
    overlap = parameters.keys() & kwargs.keys()
    if overlap:
        raise ValueError(
            f"Module `{class_name}` parameters {sorted(overlap)} are provided "
            "both in configuration and runtime arguments. Ambiguous."
        )
    parameters.update(kwargs)

    cls = mod_dict[class_name]
    try:
        return cls(**parameters)
    except Exception:
        logger.error(
            "Failed to instantiate %s with parameters: %s", cls.__name__, parameters
        )
        raise


def extract_module_parameters(
    descriptor: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Extract inline or nested parameters after removing structural fields.

    Parameters
    ----------
    descriptor : Mapping
        Provider parameters and optional ``config`` mapping. The caller must
        first remove its structural fields, such as ``name`` and ``provider``.
    context : str
        Stage description included in validation errors.

    Returns
    -------
    dict
        Independent copy of the extracted provider parameters.

    Raises
    ------
    TypeError
        If an explicit ``config`` value is not a mapping.
    ValueError
        If parameter forms are mixed or removed ``args``/``kwargs`` wrappers
        are supplied.
    """
    parameters = dict(descriptor)
    removed = parameters.keys() & {"args", "kwargs"}
    if removed:
        raise ValueError(
            f"{context}: {sorted(removed)} syntax is no longer supported; "
            "use inline parameters or `config`."
        )
    if "config" not in parameters:
        return deepcopy(parameters)

    # Presence is significant: even an empty explicit config selects nesting.
    nested = parameters.pop("config")
    if not isinstance(nested, Mapping):
        raise TypeError(f"{context} `config` must be a mapping.")
    if parameters:
        raise ValueError(f"{context} cannot mix inline parameters with `config`.")
    return deepcopy({**nested, **parameters})


def parse_module_stages(stages: Any) -> ParsedModules:
    """Normalize ordered module entries into the legacy internal representation.

    Parameters
    ----------
    stages : list of dict
        Entries with unique instance ``name``, optional implementation
        ``provider`` (defaulting to ``name``), and parameters either inline or
        in a ``config`` mapping. List order defines execution order.

    Returns
    -------
    OrderedDict
        Instance names mapped to provider ``name``, ``cfg``, and null ``priority``.

    Raises
    ------
    TypeError
        If stages, entries, or provider configurations have the wrong type.
    ValueError
        If names are invalid or duplicated, priority is supplied, or parameter
        forms are mixed. Providers validate their own parameter names.
    """
    if not isinstance(stages, list):
        raise TypeError("Module `stages` must be a list.")
    parsed: ParsedModules = OrderedDict()
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise TypeError(f"Module stage {index} must be a mapping.")
        # Ordering belongs to the list, never to priority metadata.
        if "priority" in stage:
            raise ValueError("Module stages cannot specify `priority`; use list order.")
        identities = {
            "name": stage.get("name"),
            "provider": stage.get("provider", stage.get("name")),
        }
        for field, value in identities.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Module stage {index} requires a nonempty `{field}`.")
        label = stage["name"]
        if label in parsed:
            raise ValueError(f"Duplicate module stage name `{label}`.")
        descriptor = {
            key: value
            for key, value in stage.items()
            if key not in {"name", "provider"}
        }
        config = extract_module_parameters(
            descriptor, context=f"Module stage `{label}`"
        )
        # Keep instance identity separate from the implementation name.
        parsed[label] = {
            "name": identities["provider"],
            "cfg": config,
            "priority": None,
        }
    return parsed


def parse_module_config(
    modules: Mapping[str, Any],
    name_key: str = "name",
    priority_key: str | None = "priority",
    sort_by_priority: bool = False,
    priority_descending: bool = False,
    skip_none: bool = True,
    stages_key: str | None = None,
) -> ParsedModules:
    """Parse an ordered mapping of module blocks.

    Each top-level key is treated as the module label. The implementation is
    read from ``provider``, or the deprecated ``name_key`` alias, otherwise
    the label itself is used as the provider. This supports both compact blocks such as:

    .. code-block:: yaml

        gain:
          gain: 2.0

    and repeated instances of the same module:

    .. code-block:: yaml

        first_gain:
          provider: gain
          gain: 2.0
        second_gain:
          provider: gain
          gain: 3.0

    Parameters
    ----------
    modules : Mapping
        Ordered mapping of module labels to configuration dictionaries.
    name_key : str, default 'name'
        Deprecated alias for the canonical ``provider`` implementation selector.
    priority_key : str or None, default 'priority'
        Configuration key which specifies optional execution priority.
        ``None`` leaves legacy priority fields in the provider configuration.
    sort_by_priority : bool, default False
        If ``True``, modules with smaller priority values run first. Modules
        without a priority retain their relative order after prioritized
        modules.
    priority_descending : bool, default False
        If ``True`` and ``sort_by_priority`` is ``True``, modules with larger
        priority values run first instead.
    skip_none : bool, default True
        If ``True``, skip entries explicitly set to ``None``.
    stages_key : str, optional
        Opt into an ordered-list wrapper under this key (usually ``stages``).
        It cannot be mixed with legacy module blocks. Explicit lists are never
        priority-sorted; other callers retain mapping-only behavior.

    Returns
    -------
    OrderedDict
        Mapping of module label to dictionaries with ``name``, ``cfg`` and
        ``priority`` fields.
    """
    if not isinstance(modules, Mapping):
        raise TypeError("Module configuration must be a mapping.")

    # Managers remove their own settings before sharing this format selection.
    if stages_key is not None and stages_key in modules:
        if len(modules) != 1:
            raise ValueError(f"Cannot mix `{stages_key}` with legacy module blocks.")
        return parse_module_stages(modules[stages_key])

    parsed: list[tuple[int, str, str, int | float | None, dict[str, Any]]] = []
    for index, (label, cfg) in enumerate(modules.items()):
        if cfg is None and skip_none:
            continue
        if not isinstance(cfg, Mapping):
            raise TypeError(f"Configuration for module `{label}` must be a mapping.")

        config = deepcopy(dict(cfg))
        name = resolve_module_provider(config, default=label, aliases=(name_key,))
        assert name is not None
        for selector in {"provider", name_key}:
            config.pop(selector, None)
        priority = config.pop(priority_key, None) if priority_key is not None else None
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
    modules: Mapping[str, Any],
    name_key: str = "name",
    priority_key: str | None = "priority",
    sort_by_priority: bool = False,
    priority_descending: bool = False,
    skip_none: bool = True,
    stages_key: str | None = None,
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
        Deprecated alias for the canonical ``provider`` implementation selector.
    priority_key : str or None, default 'priority'
        Configuration key which specifies optional execution priority.
        ``None`` leaves legacy priority fields in the provider configuration.
    sort_by_priority : bool, default False
        If ``True``, modules with smaller priority values run first.
    priority_descending : bool, default False
        If ``True`` and ``sort_by_priority`` is ``True``, modules with larger
        priority values run first instead.
    skip_none : bool, default True
        If ``True``, skip entries explicitly set to ``None``.
    stages_key : str, optional
        Opt into an ordered-list wrapper, as in :func:`parse_module_config`.
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
        stages_key=stages_key,
    )

    instances: OrderedDict[str, Any] = OrderedDict()
    for label, spec in parsed.items():
        cfg = dict(spec["cfg"])
        cfg["provider"] = spec["name"]
        instances[label] = instantiate(mod_dict, cfg, **kwargs)

    return instances
