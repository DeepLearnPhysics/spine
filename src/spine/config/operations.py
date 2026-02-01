"""Operations and utilities for SPINE configuration processing.

This module contains helper functions for:
- Merging dictionaries
- Parsing values
- Applying collection operations (list/dict append/remove)
- Setting nested values
- Extracting directives from configs
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import yaml

from .errors import ConfigOperationError, ConfigPathError, ConfigTypeError

__all__ = [
    "deep_merge",
    "parse_value",
    "apply_collection_operation",
    "set_nested_value",
    "extract_includes_and_overrides",
]


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge override_dict into base_dict.

    Parameters
    ----------
    base_dict : Dict[str, Any]
        Base dictionary
    override_dict : Dict[str, Any]
        Override dictionary

    Returns
    -------
    Dict[str, Any]
        Merged dictionary (new copy)
    """
    result = deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def parse_value(value_str: Any) -> Any:
    """Parse a string value into appropriate Python type.

    Parameters
    ----------
    value_str : Any
        Value to parse (if string, attempts YAML parsing)

    Returns
    -------
    Any
        Parsed value
    """
    if not isinstance(value_str, str):
        return value_str

    if value_str.strip() == "":
        return value_str

    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        return value_str


def apply_collection_operation(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
    operation: str,
    strict: str = "error",
    list_append_mode: str = "append",
) -> Dict[str, Any]:
    """Apply a collection operation (append/remove) to a nested list or dict.

    For lists:
        '+' : append values
        '-' : remove values

    For dicts:
        '-' : remove keys
        '+' : not supported

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    key_path : str
        Dot-separated path (e.g., "io.reader.file_keys")
    value : Any
        Value(s) to append/remove (single value or list)
    operation : str
        '+' (append) or '-' (remove)
    strict : str, optional
        "error" or "warn" for missing paths
    list_append_mode : str, optional
        "append" (allow duplicates) or "unique" (no duplicates)

    Returns
    -------
    Dict[str, Any]
        Modified configuration

    Raises
    ------
    ConfigPathError
        If path doesn't exist and strict="error"
    ConfigTypeError
        If target is wrong type for operation
    ConfigOperationError
        If operation is invalid
    """
    keys = key_path.split(".")
    current = config

    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            msg = f"Cannot apply collection operation to '{key_path}': path does not exist"
            if strict == "error":
                raise ConfigPathError(msg)
            warnings.warn(msg)
            return config
        elif not isinstance(current[key], dict):
            raise ConfigTypeError(
                f"Cannot apply collection operation to '{key_path}': '{key}' is not a dictionary"
            )
        current = current[key]

    final_key = keys[-1]

    # Check if target exists
    if final_key not in current:
        if operation == "+":
            # Create new list for append
            current[final_key] = []
        else:
            msg = f"Cannot remove from '{key_path}': key does not exist"
            if strict == "error":
                raise ConfigPathError(msg)
            warnings.warn(msg)
            return config

    target = current[final_key]
    values_to_process = value if isinstance(value, list) else [value]

    if isinstance(target, list):
        if operation == "+":
            if list_append_mode == "unique":
                # Add only values not already in list
                for v in values_to_process:
                    if v not in target:
                        target.append(v)
            else:
                # Append all (allow duplicates)
                current[final_key] = target + values_to_process
        elif operation == "-":
            # Remove all occurrences
            result = [item for item in target if item not in values_to_process]
            current[final_key] = result
        else:
            raise ConfigOperationError(f"Invalid collection operation: '{operation}'")

    elif isinstance(target, dict):
        if operation == "-":
            for key_to_remove in values_to_process:
                if key_to_remove in target:
                    del target[key_to_remove]
                elif strict == "warn":
                    warnings.warn(
                        f"Key '{key_to_remove}' not found in '{key_path}', skipping removal"
                    )
        elif operation == "+":
            raise ConfigOperationError(
                f"Cannot append to dict '{key_path}': '+' operation not supported for dicts"
            )
        else:
            raise ConfigOperationError(f"Invalid collection operation: '{operation}'")

    else:
        raise ConfigTypeError(
            f"Cannot apply collection operation to '{key_path}': "
            f"target is {type(target).__name__}, not a list or dict"
        )

    return config


def set_nested_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
    delete: bool = False,
    strict: str = "error",
    only_if_exists: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """Set or delete a nested value using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    key_path : str
        Dot-separated path (e.g., "io.reader.file_paths")
    value : Any
        Value to set (ignored if delete=True)
    delete : bool, optional
        If True, delete the key
    strict : str, optional
        "error" or "warn" for missing keys (when deleting)
    only_if_exists : bool, optional
        If True, only set if parent path exists

    Returns
    -------
    Tuple[Dict[str, Any], bool]
        (modified config, whether operation was applied)

    Raises
    ------
    ConfigPathError
        If strict="error" and key path doesn't exist (when deleting)
    ConfigTypeError
        If path traverses non-dict value
    """
    keys = key_path.split(".")
    current = config

    # Navigate to parent
    for i, key in enumerate(keys[:-1]):
        if key not in current:
            if delete:
                if strict == "error":
                    partial_path = ".".join(keys[: i + 1])
                    raise ConfigPathError(
                        f"Cannot delete '{key_path}': path '{partial_path}' does not exist"
                    )
                return config, False
            if only_if_exists:
                return config, False
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ConfigTypeError(
                f"Cannot set '{key_path}': '{key}' is not a dictionary"
            )
        current = current[key]

    # Set or delete final value
    final_key = keys[-1]
    if delete:
        if final_key in current:
            del current[final_key]
            return config, True
        elif strict == "error":
            raise ConfigPathError(f"Cannot delete '{key_path}': key does not exist")
        elif strict == "warn":
            warnings.warn(f"Key '{key_path}' not found, skipping deletion")
        return config, False
    else:
        current[final_key] = value
        return config, True


def extract_includes_and_overrides(
    config_dict: Any,
) -> Tuple[List[str], Dict[str, Any], List[str], Dict[str, Any]]:
    """Extract include/override/remove directives from config dict.

    Parameters
    ----------
    config_dict : Any
        Loaded YAML configuration

    Returns
    -------
    Tuple[List[str], Dict[str, Any], List[str], Dict[str, Any]]
        (includes, overrides, removals, cleaned_config)

    Raises
    ------
    ConfigOperationError
        If directive has invalid type
    """
    if not isinstance(config_dict, dict):
        return [], {}, [], config_dict

    includes = []
    overrides = {}
    removals = []
    cleaned_config = {}

    for key, value in config_dict.items():
        if key == "include":
            if isinstance(value, str):
                includes.append(value)
            elif isinstance(value, list):
                includes.extend(value)
            else:
                raise ConfigOperationError(
                    f"'include' must be a string or list of strings, got {type(value)}"
                )
        elif key == "override":
            if not isinstance(value, dict):
                raise ConfigOperationError(
                    f"'override' must be a dictionary, got {type(value)}"
                )
            overrides = value
        elif key == "remove":
            if isinstance(value, str):
                removals.append(value)
            elif isinstance(value, list):
                removals.extend(value)
            else:
                raise ConfigOperationError(
                    f"'remove' must be a string or list of strings, got {type(value)}"
                )
        else:
            cleaned_config[key] = value

    return includes, overrides, removals, cleaned_config


def _apply_overrides_and_removals(
    config: Dict[str, Any],
    overrides: Dict[str, Any],
    removals: List[str],
    strict: str,
    list_append_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply overrides and removals to config.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    overrides : Dict[str, Any]
        Override directives (may include +/- suffixes)
    removals : List[str]
        Removal directives
    strict : str
        "error" or "warn" for missing paths
    list_append_mode : str
        "append" or "unique" for list operations

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (modified config, unapplied overrides)
    """
    unapplied_overrides = {}

    # Apply removals first
    for key_path in removals:
        config, _ = set_nested_value(config, key_path, None, delete=True, strict=strict)

    # Apply overrides
    for key_path, value in overrides.items():
        parsed_value = parse_value(value)

        if key_path.endswith("+") or key_path.endswith("-"):
            # Collection operation
            base_key = key_path[:-1]
            operation = key_path[-1]
            try:
                config = apply_collection_operation(
                    config, base_key, parsed_value, operation, "error", list_append_mode
                )
            except (ConfigPathError, ConfigTypeError, ConfigOperationError) as e:
                # Check if it's a type/operation error (should be raised)
                if isinstance(e, (ConfigTypeError, ConfigOperationError)):
                    raise
                if isinstance(e, ConfigPathError) and (
                    "not a dictionary" in str(e) or "not a list" in str(e)
                ):
                    raise
                # Path doesn't exist, save for propagation
                unapplied_overrides[key_path] = value
        else:
            # Regular override
            config, applied = set_nested_value(
                config, key_path, parsed_value, only_if_exists=True
            )
            if not applied:
                unapplied_overrides[key_path] = value

    return config, unapplied_overrides
