"""Module in charge of loading SPINE configuration files.

This module provides an advanced YAML configuration loader with support for:
- File includes using the `include:` directive
- Inline includes using `!include` tags
- Dot-notation parameter overrides
- Key removal/deletion from included files
- Deep merging of configuration dictionaries

Public Functions
----------------
load_config : Load and parse a YAML configuration file
set_nested_value : Set a value in a nested dict using dot notation
parse_value : Parse a string value into appropriate Python type
deep_merge : Recursively merge two dictionaries
extract_includes_and_overrides : Extract include directives from config dict
"""

import os
from copy import deepcopy
from typing import Any, Dict, List, TextIO, Tuple, cast

import yaml

__all__ = [
    "load_config",
    "set_nested_value",
    "parse_value",
    "deep_merge",
    "extract_includes_and_overrides",
    "ConfigLoader",
]


class ConfigLoader(yaml.SafeLoader):
    """Configuration loader class.

    This class implements a more complex YAML loader than the standard loader in
    order to support more advanced functions such as:
    - Include YAML configuration files into another YAML configuration file;
    - Edit an included YAML dictionary with one liners (to modify single
      configuration parameters without replicating a configuration block).
    """

    def __init__(self, stream: TextIO) -> None:
        """Initialize the loader.

        Parameters
        ----------
        stream : TextIO
            Output of python's `open` function on a yaml file
        """
        # Fetch the parent directory where the configuration file lives
        self._root = os.path.split(stream.name)[0]

        # Initialize the base loader
        super().__init__(stream)

    def include(self, node: yaml.Node) -> Any:
        """Load and include a YAML file that is requested in the base config.

        Parameters
        ----------
        node : yaml.Node
            YAML node containing the filename to load

        Returns
        -------
        Any
            Loaded configuration dictionary
        """
        # Look for the file in the same directory as the main config file
        # construct_scalar expects a ScalarNode, but the type is validated at runtime
        filename = os.path.join(
            self._root, self.construct_scalar(cast(yaml.ScalarNode, node))
        )

        # Load the file within the base configuration
        with open(filename, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=ConfigLoader)


# Add the include constructor
ConfigLoader.add_constructor("!include", ConfigLoader.include)


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge override_dict into base_dict.

    Parameters
    ----------
    base_dict : Dict[str, Any]
        Base dictionary to merge into
    override_dict : Dict[str, Any]
        Dictionary with values to override

    Returns
    -------
    Dict[str, Any]
        Merged dictionary
    """
    result = deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def set_nested_value(
    config: Dict[str, Any], key_path: str, value: Any, delete: bool = False
) -> Dict[str, Any]:
    """Set a nested value in a dictionary using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to modify
    key_path : str
        Dot-separated path to the key (e.g., "io.reader.file_paths")
    value : Any
        Value to set (ignored if delete=True)
    delete : bool, optional
        If True, delete the key instead of setting it

    Returns
    -------
    Dict[str, Any]
        Modified configuration dictionary
    """
    keys = key_path.split(".")
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            if delete:
                # Key path doesn't exist, nothing to delete
                return config
            current[key] = {}
        elif not isinstance(current[key], dict):
            # If we encounter a non-dict value, we can't navigate further
            raise ValueError(f"Cannot set '{key_path}': '{key}' is not a dictionary")
        current = current[key]

    # Set or delete the final value
    final_key = keys[-1]
    if delete:
        if final_key in current:
            del current[final_key]
    else:
        current[final_key] = value

    return config


def extract_includes_and_overrides(
    config_dict: Any,
) -> Tuple[List[str], Dict[str, Any], List[str], Dict[str, Any]]:
    """Extract include directives, override, and remove directives from a config dict.

    Parameters
    ----------
    config_dict : Any
        Loaded YAML configuration dictionary

    Returns
    -------
    Tuple[List[str], Dict[str, Any], List[str], Dict[str, Any]]
        Tuple of (list of included files, dict of override values, list of keys to remove, cleaned config dict)
    """
    if not isinstance(config_dict, dict):
        return [], {}, [], config_dict

    includes = []
    overrides = {}
    removals = []
    cleaned_config = {}

    for key, value in config_dict.items():
        if key == "include":
            # Handle include directive
            if isinstance(value, str):
                # Single file: include: file.yaml
                includes.append(value)
            elif isinstance(value, list):
                # Multiple files: include: [file1.yaml, file2.yaml]
                includes.extend(value)
            else:
                raise ValueError(
                    f"'include' must be a string or list of strings, got {type(value)}"
                )
        elif key == "override":
            # Handle override block with dot-notation keys
            if not isinstance(value, dict):
                raise ValueError(f"'override' must be a dictionary, got {type(value)}")
            overrides = value
        elif key == "remove":
            # Handle remove directive with dot-notation keys
            if isinstance(value, str):
                # Single key: remove: io.loader.batch_size
                removals.append(value)
            elif isinstance(value, list):
                # Multiple keys: remove: [io.loader.batch_size, model.depth]
                removals.extend(value)
            else:
                raise ValueError(
                    f"'remove' must be a string or list of strings, got {type(value)}"
                )
        else:
            # Regular config key
            cleaned_config[key] = value

    return includes, overrides, removals, cleaned_config


def parse_value(value_str: Any) -> Any:
    """Parse a string value into the appropriate Python type.

    Parameters
    ----------
    value_str : Any
        String representation of the value (or any other type)

    Returns
    -------
    Any
        Parsed value
    """
    # If it's already not a string, return as-is
    if not isinstance(value_str, str):
        return value_str

    # Try to parse as YAML (handles strings, numbers, booleans, lists, etc.)
    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        # If it fails, return as string
        return value_str


def _load_config_recursive(
    cfg_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Internal function to recursively load config with accumulated overrides/removals.

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], List[str]]
        Tuple of (config dict, accumulated overrides, accumulated removals)
    """
    root_dir = os.path.dirname(os.path.abspath(cfg_path))

    # Load the YAML file (with !include support for inline includes)
    with open(cfg_path, "r", encoding="utf-8") as f:
        main_config = yaml.load(f, Loader=ConfigLoader)

    # Handle empty file
    if main_config is None:
        return {}, {}, []

    # Extract include directives, override, and removals from the loaded config
    includes, overrides, removals, cleaned_config = extract_includes_and_overrides(
        main_config
    )

    # Start with an empty config and empty accumulated overrides/removals
    config = {}
    accumulated_overrides: Dict[str, Any] = {}
    accumulated_removals: List[str] = []

    # Load all included files first (in order)
    for include_file in includes:
        include_path = os.path.join(root_dir, include_file)
        if not os.path.exists(include_path):
            raise FileNotFoundError(f"Included file not found: {include_path}")

        # Recursively load the included file
        included_config, included_overrides, included_removals = _load_config_recursive(
            include_path
        )

        # Merge the config
        config = deep_merge(config, included_config)

        # Accumulate overrides and removals from included files
        accumulated_overrides.update(included_overrides)
        accumulated_removals.extend(included_removals)

    # Merge the main config (without include/override/remove directives)
    if cleaned_config:
        config = deep_merge(config, cleaned_config)

    # Accumulate this file's overrides and removals (don't apply them yet)
    accumulated_overrides.update(overrides)
    accumulated_removals.extend(removals)

    return config, accumulated_overrides, accumulated_removals


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load a configuration file to a dictionary.

    This function supports:
    - Including other YAML files: "include: base.yaml" or "include: [base.yaml, other.yaml]"
    - Including files within blocks: "key: !include file.yaml"
    - Overriding nested parameters: "override: { io.reader.file_paths: value }"
    - Removing keys from included files: "remove: io.loader.batch_size" or "override: { key: null }"

    When files are included, their override and remove directives are accumulated
    and applied after all configs are merged, ensuring that deletions from included
    files are respected.

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file

    Returns
    -------
    Dict[str, Any]
        Loaded and merged configuration dictionary
    """
    # Load the config with accumulated overrides/removals
    config, accumulated_overrides, accumulated_removals = _load_config_recursive(
        cfg_path
    )

    # Apply all accumulated removals first (from the remove: directives)
    for key_path in accumulated_removals:
        config = set_nested_value(config, key_path, None, delete=True)

    # Apply all accumulated overrides using dot notation
    # If value is None/null, treat it as a removal
    for key_path, value in accumulated_overrides.items():
        parsed_value = parse_value(value)
        if parsed_value is None:
            # null value means delete the key
            config = set_nested_value(config, key_path, None, delete=True)
        else:
            config = set_nested_value(config, key_path, parsed_value)

    return config
