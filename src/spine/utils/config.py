"""Module in charge of loading SPINE configuration files.

This module provides an advanced YAML configuration loader with support for:
- File includes using the `include:` directive
- Inline includes using `!include` tags
- Dot-notation parameter overrides
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
    config: Dict[str, Any], key_path: str, value: Any
) -> Dict[str, Any]:
    """Set a nested value in a dictionary using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to modify
    key_path : str
        Dot-separated path to the key (e.g., "io.reader.file_paths")
    value : Any
        Value to set

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
            current[key] = {}
        elif not isinstance(current[key], dict):
            # If we encounter a non-dict value, we can't navigate further
            raise ValueError(f"Cannot set '{key_path}': '{key}' is not a dictionary")
        current = current[key]

    # Set the final value
    current[keys[-1]] = value

    return config


def extract_includes_and_overrides(
    config_dict: Any,
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """Extract include directives and overrides block from a config dict.

    Parameters
    ----------
    config_dict : Any
        Loaded YAML configuration dictionary

    Returns
    -------
    Tuple[List[str], Dict[str, Any], Dict[str, Any]]
        Tuple of (list of included files, dict of overrides, cleaned config dict)
    """
    if not isinstance(config_dict, dict):
        return [], {}, config_dict

    includes = []
    overrides = {}
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
        elif key == "overrides":
            # Handle overrides block with dot-notation keys
            if not isinstance(value, dict):
                raise ValueError(f"'overrides' must be a dictionary, got {type(value)}")
            overrides = value
        else:
            # Regular config key
            cleaned_config[key] = value

    return includes, overrides, cleaned_config


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


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load a configuration file to a dictionary.

    This function supports:
    - Including other YAML files: "include: base.yaml" or "include: [base.yaml, other.yaml]"
    - Including files within blocks: "key: !include file.yaml"
    - Overriding nested parameters: "overrides: { io.reader.file_paths: value }"

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file

    Returns
    -------
    Dict[str, Any]
        Loaded and merged configuration dictionary
    """
    root_dir = os.path.dirname(os.path.abspath(cfg_path))

    # Load the YAML file (with !include support for inline includes)
    with open(cfg_path, "r", encoding="utf-8") as f:
        main_config = yaml.load(f, Loader=ConfigLoader)

    # Handle empty file
    if main_config is None:
        return {}

    # Extract include directives and overrides from the loaded config
    includes, overrides, cleaned_config = extract_includes_and_overrides(main_config)

    # Start with an empty config
    config = {}

    # Load all included files first (in order)
    for include_file in includes:
        include_path = os.path.join(root_dir, include_file)
        if not os.path.exists(include_path):
            raise FileNotFoundError(f"Included file not found: {include_path}")

        # Recursively load the included file (supports nested includes)
        included_config = load_config(include_path)
        config = deep_merge(config, included_config)

    # Merge the main config (without include/override directives)
    if cleaned_config:
        config = deep_merge(config, cleaned_config)

    # Apply overrides using dot notation
    for key_path, value in overrides.items():
        parsed_value = parse_value(value)
        config = set_nested_value(config, key_path, parsed_value)

    return config
