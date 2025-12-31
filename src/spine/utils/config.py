"""Module in charge of loading SPINE configuration files."""

import os
import re
from copy import deepcopy

import yaml


class ConfigLoader(yaml.SafeLoader):
    """Configuration loader class.

    This class implements a more complex YAML loader than the standard loader in
    order to support more advanced functions such as:
    - Include YAML configuration files into another YAML configuration file;
    - Edit an included YAML dictionary with one liners (to modify single
      configuration parameters without replicating a configuration block).
    """

    def __init__(self, stream):
        """Initialize the loader.

        Parameters
        ----------
        stream : _io.TextIOWrapper
            Output of python's `open` function on a yaml file
        """
        # Fetch the parent directory where the configuration file lives
        self._root = os.path.split(stream.name)[0]

        # Initialize the base loader
        super().__init__(stream)

    def include(self, node):
        """Load and include a YAML file that is requested in the base config.

        Parameters
        ----------
        node : str
            Name of the YAML block to load
        """
        # Look for the file in the same directory as the main config file
        filename = os.path.join(self._root, self.construct_scalar(node))

        # Load the file within the base configuration
        with open(filename, "r") as f:
            return yaml.load(f, Loader=ConfigLoader)


# Add the include constructor
ConfigLoader.add_constructor("!include", ConfigLoader.include)


def _deep_merge(base_dict, override_dict):
    """Recursively merge override_dict into base_dict.

    Parameters
    ----------
    base_dict : dict
        Base dictionary to merge into
    override_dict : dict
        Dictionary with values to override

    Returns
    -------
    dict
        Merged dictionary
    """
    result = deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _set_nested_value(config, key_path, value):
    """Set a nested value in a dictionary using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary to modify
    key_path : str
        Dot-separated path to the key (e.g., "io.reader.file_paths")
    value : any
        Value to set

    Returns
    -------
    dict
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


def _extract_includes_and_overrides(config_dict):
    """Extract include directives and dot-notation overrides from a config dict.

    Parameters
    ----------
    config_dict : dict
        Loaded YAML configuration dictionary

    Returns
    -------
    tuple
        (list of included files, dict of overrides, cleaned config dict)
    """
    if not isinstance(config_dict, dict):
        return [], {}, config_dict

    includes = []
    overrides = {}
    cleaned_config = {}

    # Pattern to match: "key.path.here" for dot notation keys
    dotted_key_pattern = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+$"
    )

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
        elif dotted_key_pattern.match(key):
            # This is a dot-notation override
            overrides[key] = value
        else:
            # Regular config key
            cleaned_config[key] = value

    return includes, overrides, cleaned_config


def _parse_value(value_str):
    """Parse a string value into the appropriate Python type.

    Parameters
    ----------
    value_str : str
        String representation of the value

    Returns
    -------
    any
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


def load_config(cfg_path):
    """Load a configuration file to a dictionary.

    This function supports:
    - Including other YAML files: "include: base.yaml" or "include: [base.yaml, other.yaml]"
    - Including files within blocks: "key: !include file.yaml"
    - Overriding nested parameters with dot notation: "io.reader.file_paths: value"

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file

    Returns
    -------
    dict
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
    includes, overrides, cleaned_config = _extract_includes_and_overrides(main_config)

    # Start with an empty config
    config = {}

    # Load all included files first (in order)
    for include_file in includes:
        include_path = os.path.join(root_dir, include_file)
        if not os.path.exists(include_path):
            raise FileNotFoundError(f"Included file not found: {include_path}")

        # Recursively load the included file (supports nested includes)
        included_config = load_config(include_path)
        config = _deep_merge(config, included_config)

    # Merge the main config (without include/override directives)
    if cleaned_config:
        config = _deep_merge(config, cleaned_config)

    # Apply overrides using dot notation
    for key_path, value in overrides.items():
        parsed_value = _parse_value(value)
        config = _set_nested_value(config, key_path, parsed_value)

    return config
