"""Main configuration loading functions for SPINE.

This module provides the primary entry points for loading SPINE configurations:
- load_config(): Load from a YAML string
- load_config_file(): Load from a file path
- _load_config_recursive(): Internal recursive loader with include support
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .api import META_DESCRIPTION, META_KEY, META_LIST_APPEND, META_STRICT, META_VERSION
from .errors import ConfigCycleError, ConfigIncludeError
from .loader import ConfigLoader, resolve_config_path
from .meta import check_compatibility, extract_metadata
from .operations import (
    _apply_overrides_and_removals,
    apply_collection_operation,
    deep_merge,
    extract_includes_and_overrides,
    parse_value,
    set_nested_value,
)

__all__ = ["load_config", "load_config_file"]


def _load_config_recursive(
    cfg_path: Optional[str] = None,
    config_string: Optional[str] = None,
    root_dir: Optional[str] = None,
    include_stack: Optional[List[str]] = None,
    compatibility_checks: Optional[List[Tuple[Dict, Dict, str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """Recursively load config with cycle detection.

    Parameters
    ----------
    cfg_path : Optional[str]
        Path to configuration file (mutually exclusive with config_string)
    config_string : Optional[str]
        YAML configuration string (mutually exclusive with cfg_path)
    root_dir : Optional[str]
        Root directory for resolving relative include paths.
        Required when using config_string with includes.
        Defaults to directory of cfg_path when loading from file.
    include_stack : Optional[List[str]]
        Stack of currently-loading files (for cycle detection)
    compatibility_checks : Optional[List[Tuple[Dict, Dict, str]]]
        List to accumulate (parent_meta, included_meta, path) for deferred checking

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]
        (config content, override directives, removal directives, metadata)

    Raises
    ------
    ConfigCycleError
        If circular include detected
    ConfigIncludeError
        If included file not found
    ValueError
        If both or neither cfg_path and config_string are provided
    """
    # Validate inputs
    if (cfg_path is None) == (config_string is None):
        raise ValueError("Must provide exactly one of cfg_path or config_string")

    # Determine the identifier for cycle detection and root directory
    if cfg_path is not None:
        cfg_path = os.path.abspath(cfg_path)
        identifier = cfg_path
        if root_dir is None:
            root_dir = os.path.dirname(cfg_path)
    else:
        # For string configs, use a pseudo-identifier
        identifier = "<string>"
        if root_dir is None:
            root_dir = os.getcwd()

    # Cycle detection
    if include_stack is None:
        include_stack = []
    if compatibility_checks is None:
        compatibility_checks = []

    if identifier in include_stack and cfg_path is not None:
        cycle = include_stack + [identifier]
        raise ConfigCycleError(cycle)

    include_stack = include_stack + [identifier]

    # Create a custom loader class with the specified root_dir
    class CustomConfigLoader(ConfigLoader):
        def __init__(self, stream):
            super().__init__(stream, root_dir)

    # Load YAML
    try:
        if cfg_path is not None:
            with open(cfg_path, "r", encoding="utf-8") as f:
                main_config = yaml.load(f, Loader=CustomConfigLoader)
        else:
            assert config_string is not None  # For the linter's sake
            main_config = yaml.load(config_string, Loader=CustomConfigLoader)
    except FileNotFoundError as exc:
        raise ConfigIncludeError(f"Configuration file not found: {cfg_path}") from exc
    except Exception as exc:
        source = cfg_path if cfg_path else "<string>"
        raise ConfigIncludeError(f"Error loading {source}: {exc}") from exc

    if main_config is None:
        return {}, {}, [], {}

    # Extract metadata
    metadata = extract_metadata(main_config, cfg_path if cfg_path else "<string>")
    strict = metadata[META_STRICT]
    list_append_mode = metadata[META_LIST_APPEND]

    # Extract directives
    includes, overrides, removals, cleaned_config = extract_includes_and_overrides(
        main_config
    )

    # Remove __meta__ from cleaned config
    if META_KEY in cleaned_config:
        del cleaned_config[META_KEY]

    config = {}

    # Process includes
    for include_file in includes:
        # Resolve include path with SPINE_CONFIG_PATH support
        include_path = resolve_config_path(include_file, root_dir)

        # Recursively load
        (
            included_config,
            included_overrides,
            included_removals,
            included_meta,
        ) = _load_config_recursive(
            cfg_path=include_path,
            root_dir=None,
            include_stack=include_stack,
            compatibility_checks=compatibility_checks,
        )

        # Warn if included file has no metadata (but keep the metadata that was extracted)
        if not included_meta.get(META_VERSION) and not included_meta.get(
            META_DESCRIPTION
        ):
            # File likely has no __meta__ block (only defaults)
            file_name = os.path.basename(include_path)
            warnings.warn(
                f"Included file '{file_name}' has no __meta__ block. "
                f"Consider adding metadata for better configuration management.",
                stacklevel=2,
            )

        # Defer compatibility check until all includes loaded
        compatibility_checks.append((metadata, included_meta, include_path))

        # Merge included config
        config = deep_merge(config, included_config)

        # Merge component versions from included metadata into parent
        # This allows subsequent includes to check against accumulated components
        if "components" in included_meta:
            if "components" not in metadata:
                metadata["components"] = {}
            metadata["components"].update(included_meta["components"])
        elif META_VERSION in included_meta:
            # If included file has version but no components, infer component name from file path
            # e.g., base/base_240719.yaml -> component "base" with version "240719"
            # This allows configs without explicit components to still participate in version checking
            include_dir = os.path.basename(os.path.dirname(include_path))
            if include_dir and include_dir not in ("", "."):
                if "components" not in metadata:
                    metadata["components"] = {}
                metadata["components"][include_dir] = included_meta[META_VERSION]

        # Apply included overrides (use included file's strict/list_append settings)
        included_strict = included_meta.get(META_STRICT, strict)
        included_list_append = included_meta.get(META_LIST_APPEND, list_append_mode)

        config, unapplied = _apply_overrides_and_removals(
            config,
            included_overrides,
            included_removals,
            included_strict,
            included_list_append,
        )

        # Propagate unapplied overrides
        if unapplied:
            overrides = {**unapplied, **overrides}

    # Merge main config content
    if cleaned_config:
        config = deep_merge(config, cleaned_config)

    return config, overrides, removals, metadata


def load_config(config_str: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a SPINE configuration from a YAML string.

    Similar to yaml.safe_load(), but with SPINE's advanced features:
    - Hierarchical includes with cycle detection
    - Metadata via __meta__ blocks
    - Override semantics with dot-notation
    - Collection operations (list append/remove, dict key removal)
    - Configurable strict modes (warn/error)

    See module docstring for full configuration language spec.

    Parameters
    ----------
    config_str : str
        YAML configuration string
    root_dir : Optional[str]
        Root directory for resolving relative include paths.
        Also used as the base for SPINE_CONFIG_PATH searches.
        If not provided, defaults to current working directory.
        Required if config contains __include__ directives with relative paths.

    Returns
    -------
    Dict[str, Any]
        Loaded and merged configuration

    Raises
    ------
    ConfigCycleError
        If circular include detected
    ConfigIncludeError
        If included file not found or can't be loaded
    ConfigPathError
        If removal/operation targets non-existent path (when strict="error")
    ConfigTypeError
        If operation applied to wrong type
    ConfigOperationError
        If invalid operation specified

    Examples
    --------
    Simple string config:

    >>> config_str = \"\"\"
    ... io:
    ...   reader:
    ...     batch_size: 32
    ... \"\"\"
    >>> config = load_config(config_str)
    >>> print(config['io']['reader']['batch_size'])
    32

    String config with includes (requires root_dir):

    >>> config_str = \"\"\"
    ... include: base.yaml
    ... model:
    ...   name: resnet
    ... \"\"\"
    >>> config = load_config(config_str, root_dir="/path/to/configs")

    For loading from files, use load_config_file():

    >>> config = load_config_file("config.yaml")

    Or equivalently:

    >>> with open("config.yaml") as f:
    ...     config = load_config(f.read())

    See Also
    --------
    load_config_file : Load configuration from a file path
    """
    # Load recursively, accumulating compatibility checks
    compatibility_checks = []
    config, overrides, removals, metadata = _load_config_recursive(
        config_string=config_str,
        root_dir=root_dir,
        compatibility_checks=compatibility_checks,
    )

    # Now that all includes are loaded, check all compatibility requirements
    for parent_meta, included_meta, include_path in compatibility_checks:
        check_compatibility(parent_meta, included_meta, include_path)

    # Get strict and list_append settings from top-level metadata
    # (these are always present, set by extract_metadata with defaults)
    strict = metadata[META_STRICT]
    list_append_mode = metadata[META_LIST_APPEND]

    # Apply top-level overrides
    # Note: these include both explicit top-level overrides and propagated ones from nested files
    # Use strict mode from top-level metadata
    for key_path, value in overrides.items():
        parsed_value = parse_value(value)

        if key_path.endswith("+") or key_path.endswith("-"):
            # Collection operations - use strict mode from metadata
            base_key = key_path[:-1]
            operation = key_path[-1]
            config = apply_collection_operation(
                config, base_key, parsed_value, operation, strict, list_append_mode
            )
        else:
            # Regular override - silently skip if parent doesn't exist
            config, _ = set_nested_value(
                config, key_path, parsed_value, only_if_exists=True
            )

    # Apply top-level removals
    for key_path in removals:
        config, _ = set_nested_value(config, key_path, None, delete=True, strict=strict)

    # Remove __meta__ from final config
    if META_KEY in config:
        del config[META_KEY]

    return config


def load_config_file(cfg_path: str) -> Dict[str, Any]:
    """Load a SPINE configuration from a file.

    Convenience function that reads a configuration file and passes it to load_config.
    The file's directory is automatically used as root_dir for include resolution.

    Parameters
    ----------
    cfg_path : str
        Path to configuration file

    Returns
    -------
    Dict[str, Any]
        Loaded and merged configuration

    Raises
    ------
    ConfigCycleError
        If circular include detected
    ConfigIncludeError
        If included file not found or can't be loaded
    ConfigPathError
        If removal/operation targets non-existent path (when strict="error")
    ConfigTypeError
        If operation applied to wrong type
    ConfigOperationError
        If invalid operation specified

    Examples
    --------
    >>> config = load_config_file("config.yaml")
    >>> print(config['io']['reader']['batch_size'])
    32

    See Also
    --------
    load_config : Load config from a YAML string
    """
    # Load recursively, accumulating compatibility checks
    compatibility_checks = []
    cfg_path = os.path.abspath(cfg_path)
    root_dir = os.path.dirname(cfg_path)

    config, overrides, removals, metadata = _load_config_recursive(
        cfg_path=cfg_path, root_dir=root_dir, compatibility_checks=compatibility_checks
    )

    # Now that all includes are loaded, check all compatibility requirements
    for parent_meta, included_meta, include_path in compatibility_checks:
        check_compatibility(parent_meta, included_meta, include_path)

    # Get strict and list_append settings from top-level metadata
    # (these are always present, set by extract_metadata with defaults)
    strict = metadata[META_STRICT]
    list_append_mode = metadata[META_LIST_APPEND]

    # Apply top-level overrides
    # Note: these include both explicit top-level overrides and propagated ones from nested files
    # Use strict mode from top-level metadata
    for key_path, value in overrides.items():
        parsed_value = parse_value(value)

        if key_path.endswith("+") or key_path.endswith("-"):
            # Collection operations - use strict mode from metadata
            base_key = key_path[:-1]
            operation = key_path[-1]
            config = apply_collection_operation(
                config, base_key, parsed_value, operation, strict, list_append_mode
            )
        else:
            # Regular override - silently skip if parent doesn't exist
            config, _ = set_nested_value(
                config, key_path, parsed_value, only_if_exists=True
            )

    # Apply top-level removals
    for key_path in removals:
        config, _ = set_nested_value(config, key_path, None, delete=True, strict=strict)

    # Remove __meta__ from final config
    if META_KEY in config:
        del config[META_KEY]

    return config
