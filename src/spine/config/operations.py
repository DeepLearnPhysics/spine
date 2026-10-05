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
from dataclasses import dataclass
from os.path import expandvars
from typing import Any, Dict, List, Tuple

import yaml

from .errors import (
    ConfigOperationError,
    ConfigPathError,
    ConfigTypeError,
    ConfigValidationError,
)

__all__ = [
    "deep_merge",
    "expand_env_vars",
    "parse_value",
    "apply_overrides",
    "apply_collection_operation",
    "apply_named_list_edits",
    "apply_overrides_and_removals",
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


def apply_overrides(
    config: Dict[str, Any], overrides: List[str] | None
) -> Dict[str, Any]:
    """Apply serialized configuration overrides using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    overrides : List[str], optional
        Overrides written as ``key.path=value``

    Returns
    -------
    Dict[str, Any]
        Modified configuration

    Raises
    ------
    ValueError
        If an override does not follow the ``key.path=value`` format
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: '{override}'. "
                f"Expected format: 'key.path=value'"
            )

        key_path, value_str = override.split("=", 1)

        # Parse basic scalars and collections before updating the nested key.
        value = parse_value(value_str.strip())
        key_path = key_path.strip()
        if key_path.endswith(("+", "-", "~")):
            config = apply_collection_operation(
                config, key_path[:-1], value, key_path[-1]
            )
        else:
            config, _ = set_nested_value(config, key_path, value)

    return config


def expand_env_vars(value: Any) -> Any:
    """Recursively expand shell environment variables in string config values.

    Parameters
    ----------
    value : Any
        Value to expand (string, list, dict, or other)

    Returns
    -------
    Any
        Value with environment variables expanded (if string) or recursively processed
    """
    if isinstance(value, str):
        return expandvars(value)

    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]

    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}

    return value


def apply_named_list_edits(
    config: Dict[str, Any], key_path: str, edits: Any
) -> Dict[str, Any]:
    """Apply ordered edits to a list of uniquely named mappings.

    Parameters
    ----------
    config : dict
        Configuration to update in place after all edits succeed.
    key_path : str
        Dot-separated path to an existing list.
    edits : dict or list of dict
        One edit or an ordered sequence of edits. Each edit contains exactly
        one of ``insert``, ``update``, or ``remove``. Insertion takes ``value``
        and exactly one named ``before``/``after`` anchor. Update takes ``name``
        and ``changes``; removal takes ``name``.

    Returns
    -------
    dict
        Configuration with the edited list. Other values are unchanged.

    Raises
    ------
    ConfigPathError
        If the list path or a named target does not exist.
    ConfigTypeError
        If the target is not a list or a parent is not a mapping.
    ConfigOperationError
        If an edit is malformed or names are missing, invalid, or duplicated.

    Notes
    -----
    Updates recursively merge mappings and replace other values, including
    lists and nulls. Names cannot be updated. Failures never partially apply
    an edit sequence, and no edits are deferred to a later include.
    """
    # Resolve the existing list without creating missing parent mappings.
    if not key_path or any(not key for key in key_path.split(".")):
        raise ConfigPathError(f"Invalid named-list path '{key_path}'.")
    current = config
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current:
            raise ConfigPathError(f"Named-list path '{key_path}' does not exist.")
        if not isinstance(current[key], dict):
            raise ConfigTypeError(
                f"Named-list path '{key_path}': '{key}' is not a mapping."
            )
        current = current[key]
    if keys[-1] not in current:
        raise ConfigPathError(f"Named-list path '{key_path}' does not exist.")
    target = current[keys[-1]]
    if not isinstance(target, list):
        raise ConfigTypeError(f"Named-list path '{key_path}' must contain a list.")

    def valid_name(value: Any) -> bool:
        """Check that an identity is a nonempty string."""
        return isinstance(value, str) and bool(value.strip())

    # Validate every entry so named targets are always unambiguous.
    names = set()
    for entry in target:
        if not isinstance(entry, dict) or not valid_name(entry.get("name")):
            raise ConfigOperationError(
                f"Named-list path '{key_path}' requires mappings with nonempty names."
            )
        if entry["name"] in names:
            raise ConfigOperationError(
                f"Named-list path '{key_path}' has duplicate name '{entry['name']}'."
            )
        names.add(entry["name"])

    # Normalize single-edit shorthand, then work on a copy until all edits pass.
    if isinstance(edits, dict):
        edits = [edits]
    if not isinstance(edits, list) or not edits:
        raise ConfigOperationError(
            f"Named-list edits for '{key_path}' must be a mapping or nonempty list."
        )
    result = deepcopy(target)
    for index, edit in enumerate(edits):
        context = f"Named-list edit {index + 1} for '{key_path}'"
        if not isinstance(edit, dict) or len(edit) != 1:
            raise ConfigOperationError(f"{context} requires exactly one operation.")
        operation, options = next(iter(edit.items()))
        if operation not in {"insert", "update", "remove"}:
            raise ConfigOperationError(f"{context}: unknown operation '{operation}'.")
        if not isinstance(options, dict):
            raise ConfigOperationError(f"{context}: options must be a mapping.")

        # Validate each operation's fields and identify its anchor or target.
        if operation == "insert":
            anchors = set(options) & {"before", "after"}
            if len(anchors) != 1 or set(options) != anchors | {"value"}:
                raise ConfigOperationError(
                    f"{context}: insert requires value and exactly one of before/after."
                )
            anchor = next(iter(anchors))
            name = options[anchor]
            value = options["value"]
            if not isinstance(value, dict) or not valid_name(value.get("name")):
                raise ConfigOperationError(
                    f"{context}: inserted value requires a nonempty name."
                )
            if value["name"] in names:
                raise ConfigOperationError(
                    f"{context}: duplicate inserted name '{value['name']}'."
                )
        else:
            expected = {"name", "changes"} if operation == "update" else {"name"}
            if set(options) != expected:
                raise ConfigOperationError(
                    f"{context}: {operation} requires only {sorted(expected)}."
                )
            name = options["name"]
            if operation == "update":
                changes = options["changes"]
                if not isinstance(changes, dict) or "name" in changes:
                    raise ConfigOperationError(
                        f"{context}: changes must be a mapping without 'name'."
                    )

        # Resolve against the current result, including all preceding edits.
        if not valid_name(name):
            raise ConfigOperationError(f"{context}: target name must be nonempty.")
        if name not in names:
            raise ConfigPathError(f"{context}: target '{name}' does not exist.")
        position = next(i for i, entry in enumerate(result) if entry["name"] == name)
        if operation == "insert":
            value = options["value"]
            result.insert(position + ("after" in options), deepcopy(value))
            names.add(value["name"])
        elif operation == "update":
            result[position] = deep_merge(
                result[position], deepcopy(options["changes"])
            )
        else:
            result.pop(position)
            names.remove(name)

    # Publish only the completed sequence; failures leave the original intact.
    current[keys[-1]] = result
    return config


def apply_collection_operation(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
    operation: str,
    strict: str = "error",
    list_append_mode: str = "append",
) -> Dict[str, Any]:
    """Apply a collection operation to a nested list or dict.

    For lists:
        '+' : append values
        '-' : remove values
        '~' : edit entries by unique name

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
        '+' (append), '-' (remove), or '~' (named-list edits)
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
    if operation == "~":
        return apply_named_list_edits(config, key_path, value)

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
                warnings.warn(f"Cannot delete '{key_path}': parent path does not exist")
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


@dataclass(frozen=True)
class ConfigDirective:
    """One ordered configuration operation, with its declaring file's policy.

    Attributes
    ----------
    path : str
        Dotted target path, without an operator suffix.
    value : Any
        Operation payload.
    operation : str
        Assignment (=), collection operator, or explicit path removal (remove).
    source : str
        File that declared the operation.
    strict : str
        Missing-target policy for collections and explicit removals.
    list_append_mode : str
        Append or unique-list behavior from the declaring file.
    optional : bool
        Whether an absent target may be skipped at final resolution.
    """

    path: str
    value: Any
    operation: str
    source: str
    strict: str
    list_append_mode: str
    optional: bool = False


def make_config_directives(
    overrides: Dict[str, Any],
    removals: List[str],
    strict: str,
    list_append_mode: str,
    *,
    source: str = "<config>",
    optional_paths: List[str] | None = None,
) -> List[ConfigDirective]:
    """Capture declaration order and policy before entering an include context.

    Parameters
    ----------
    overrides : dict
        Override entries in declaration order.
    removals : list of str
        Explicit path deletions, applied after overrides.
    strict : str
        Missing-target policy.
    list_append_mode : str
        Collection append policy.
    source : str, optional
        Declaring file used in diagnostics.
    optional_paths : list of str, optional
        Exact target paths allowed to be absent in this file's directives.
    """
    optional = set(optional_paths or [])
    directives = []
    for key, value in overrides.items():
        operation = key[-1] if key.endswith(("+", "-", "~")) else "="
        path = key[:-1] if operation != "=" else key
        if operation == "~" and path in optional:
            raise ConfigValidationError(
                f"{source}: named-list edits at '{path}' cannot be optional."
            )
        directives.append(
            ConfigDirective(
                path,
                value,
                operation,
                source,
                strict,
                list_append_mode,
                path in optional,
            )
        )
    directives.extend(
        ConfigDirective(
            path, None, "remove", source, strict, list_append_mode, path in optional
        )
        for path in removals
    )
    unused = optional - {directive.path for directive in directives}
    if unused:
        raise ConfigValidationError(
            f"{source}: optional_paths do not match local directives: {sorted(unused)}."
        )
    return directives


def _paths_overlap(left: str, right: str) -> bool:
    """Whether either operation can affect the other operation's target."""
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


def _apply_config_directives(
    config: Dict[str, Any], directives: List[ConfigDirective], defer_missing: bool
) -> Tuple[Dict[str, Any], List[ConfigDirective]]:
    """Apply available operations without letting overlapping ones overtake."""
    pending: List[ConfigDirective] = []
    for directive in directives:
        path = directive.path
        operation = directive.operation
        # Preserve dependencies while letting unrelated configuration proceed.
        if any(_paths_overlap(path, previous.path) for previous in pending):
            if operation == "~":
                raise ConfigPathError(
                    f"{directive.source}: named-list edit '{path}' is blocked by "
                    "an unresolved earlier operation."
                )
            pending.append(directive)
            continue

        try:
            if operation == "=":
                config, applied = set_nested_value(
                    config, path, parse_value(directive.value), only_if_exists=True
                )
                if not applied:
                    raise ConfigPathError(
                        f"Cannot assign '{path}': parent path does not exist"
                    )
            elif operation == "remove":
                config, _ = set_nested_value(
                    config, path, None, delete=True, strict="error"
                )
            else:
                config = apply_collection_operation(
                    config,
                    path,
                    parse_value(directive.value),
                    operation,
                    "error",
                    directive.list_append_mode,
                )
        except ConfigPathError as exc:
            message = f"{directive.source}: {exc}"
            # Named edits and malformed paths are never optional or deferred.
            if (
                operation == "~"
                or "not a dictionary" in str(exc)
                or "not a list" in str(exc)
            ):
                raise ConfigPathError(message) from exc
            if defer_missing and operation != "remove":
                pending.append(directive)
            elif directive.optional:
                continue
            elif operation == "=":
                warnings.warn(
                    message
                    + ". Silently skipping unresolved assignments is deprecated; "
                    "declare this path in __meta__.optional_paths if it is optional. "
                    f"A future release will honor strict: {directive.strict}.",
                    FutureWarning,
                    stacklevel=3,
                )
            elif directive.strict == "warn":
                warnings.warn(message, stacklevel=3)
            else:
                raise ConfigPathError(message) from exc
        except (ConfigTypeError, ConfigOperationError) as exc:
            raise type(exc)(f"{directive.source}: {exc}") from exc
    return config, pending


def apply_overrides_and_removals(
    config: Dict[str, Any],
    overrides: Dict[str, Any] | List[ConfigDirective],
    removals: List[str],
    strict: str,
    list_append_mode: str,
    *,
    defer_missing: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any] | List[ConfigDirective]]:
    """Apply ordered directives, preserving deferred operations and provenance.

    Parameters
    ----------
    config : dict
        Configuration to modify.
    overrides : dict or list of ConfigDirective
        Public mapping input or the loader's ordered, source-aware operations.
    removals : list of str
        Explicit deletions after overrides. Internal sequences include these.
    strict : str
        Missing-target policy for mapping input.
    list_append_mode : str
        Append policy for mapping input.
    defer_missing : bool, default True
        Retain unresolved operations until an enclosing configuration supplies
        their targets. Final loading boundaries warn or fail instead.

    Returns
    -------
    tuple
        Updated config and pending operations. Mapping inputs retain mapping
        output unless a deferred removal requires the ordered representation.
    """
    is_mapping = isinstance(overrides, dict)
    directives = (
        make_config_directives(overrides, removals, strict, list_append_mode)
        if is_mapping
        else overrides
    )
    config, pending = _apply_config_directives(config, directives, defer_missing)
    if is_mapping and all(item.operation != "remove" for item in pending):
        return config, {
            directive.path
            + (
                directive.operation if directive.operation in {"+", "-", "~"} else ""
            ): directive.value
            for directive in pending
        }
    return config, pending
