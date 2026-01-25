"""SPINE configuration loader with advanced features.

This module provides a sophisticated YAML configuration loading system with:

- **File inclusion**: Hierarchical includes with cycle detection
- **Metadata support**: __meta__ blocks for versioning and configuration
- **Override semantics**: Dot-notation overrides with propagation
- **Collection operations**: List append/remove, dict key removal
- **Strict modes**: Configurable warn/error behavior for missing paths
- **Type safety**: Typed exceptions for better error handling
- **File downloads**: Automatic downloading and caching of remote files

Configuration Language (API v1.0)
----------------------------------

Include Semantics:
    include: base.yaml               # Single file
    include: [base.yaml, other.yaml] # Multiple files (order matters)
    key: !include inline.yaml        # Inline include

Path Resolution:
    model:
      weights: !path weights/model.ckpt  # Resolve relative path
    post:
      cfg: !path config.yaml             # Must exist at load time

File Downloads:
    model:
      # Simple URL download
      weights: !download https://example.com/model.ckpt

      # With hash validation (recommended for production)
      weights: !download
        url: https://example.com/model.ckpt
        hash: abc123...  # SHA256 hash for validation

    Downloaded files are cached in weights/ (or $SPINE_CACHE_DIR).
    Files are only downloaded once and reused on subsequent loads.

Override Semantics:
    override:
      io.reader.batch_size: 32       # Set value
      io.reader.file_paths: null     # Set to None (not delete)
      parsers+: [new_parser]         # Append to list
      parsers-: [old_parser]         # Remove from list
      io.writer-: [key1, key2]       # Remove dict keys

Removal Semantics:
    remove: io.loader.batch_size     # Delete key (strict by default)
    remove: [key1, key2]             # Delete multiple keys

Metadata:
    __meta__:
      kind: mod                      # "mod" or "bundle"
      strict: warn                   # "warn" or "error" for missing removals
      list_append: unique            # "unique" or "append" for list ops
      version: "1.0"
      date: "2024-01-01"
      description: "Config description"

Application Order:
    1. Load and merge all included files recursively (depth-first)
    2. Apply each included file's overrides immediately after merging
    3. Unapplied overrides (missing paths) propagate upward
    4. Parent config content is merged
    5. Parent overrides applied at end

Public Functions
----------------
load_config : Load and parse a YAML configuration file (main entry point)
"""

import os
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, TextIO, Tuple, cast

import yaml

from .api import META_DESCRIPTION, META_KEY, META_LIST_APPEND, META_STRICT, META_VERSION
from .download import download_from_url
from .errors import (
    ConfigCycleError,
    ConfigIncludeError,
    ConfigOperationError,
    ConfigPathError,
    ConfigTypeError,
)
from .meta import check_compatibility, extract_metadata

__all__ = [
    "load_config",
    "set_nested_value",
    "parse_value",
    "deep_merge",
    "extract_includes_and_overrides",
    "apply_collection_operation",
    "ConfigLoader",
    "resolve_config_path",
]


def resolve_config_path(
    filename: str, current_dir: str, search_paths: Optional[List[str]] = None
) -> str:
    """Resolve a configuration file path with SPINE_CONFIG_PATH support.

    Resolution order:
    1. If absolute path, return as-is
    2. Try relative to current_dir
    3. Try relative to current_dir with .yaml/.yml extension
    4. Search through SPINE_CONFIG_PATH directories
    5. Search through SPINE_CONFIG_PATH directories with .yaml/.yml extension

    Parameters
    ----------
    filename : str
        Config filename or path to resolve
    current_dir : str
        Directory of the config file doing the including
    search_paths : Optional[List[str]]
        List of search paths (defaults to SPINE_CONFIG_PATH env var)

    Returns
    -------
    str
        Resolved absolute path

    Raises
    ------
    ConfigIncludeError
        If file cannot be found in any location
    """
    # If absolute path, check if it exists
    if os.path.isabs(filename):
        if os.path.exists(filename):
            return filename
        raise ConfigIncludeError(f"Absolute path not found: {filename}")

    # Try relative to current directory
    relative_path = os.path.join(current_dir, filename)
    if os.path.exists(relative_path):
        return os.path.abspath(relative_path)

    # Try with .yaml/.yml extension relative to current directory
    for ext in [".yaml", ".yml"]:
        if not filename.endswith(ext):
            relative_path_with_ext = relative_path + ext
            if os.path.exists(relative_path_with_ext):
                return os.path.abspath(relative_path_with_ext)

    # Get search paths from environment variable if not provided
    if search_paths is None:
        env_paths = os.environ.get("SPINE_CONFIG_PATH", "")
        search_paths = [p.strip() for p in env_paths.split(":") if p.strip()]

    # Search through SPINE_CONFIG_PATH
    for search_dir in search_paths:
        search_path = os.path.join(search_dir, filename)
        if os.path.exists(search_path):
            return os.path.abspath(search_path)

        # Try with extensions
        for ext in [".yaml", ".yml"]:
            if not filename.endswith(ext):
                search_path_with_ext = search_path + ext
                if os.path.exists(search_path_with_ext):
                    return os.path.abspath(search_path_with_ext)

    # File not found anywhere
    search_locations = [f"Relative to: {current_dir}"]
    if search_paths:
        search_locations.append(f"SPINE_CONFIG_PATH: {':'.join(search_paths)}")

    raise ConfigIncludeError(
        f"Config file '{filename}' not found.\n"
        f"Searched in:\n  - " + "\n  - ".join(search_locations)
    )


class ConfigLoader(yaml.SafeLoader):
    """YAML loader with !include tag support.

    This loader extends yaml.SafeLoader to support inline file includes
    using the !include tag.
    """

    def __init__(self, stream: TextIO) -> None:
        """Initialize the loader.

        Parameters
        ----------
        stream : TextIO
            File stream (from `open()`)
        """
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)

    def include(self, node: yaml.Node) -> Any:
        """Load and include a YAML file inline.

        Parameters
        ----------
        node : yaml.Node
            YAML node containing the filename

        Returns
        -------
        Any
            Loaded configuration content
        """
        filename = self.construct_scalar(cast(yaml.ScalarNode, node))
        resolved_path = resolve_config_path(filename, self._root)

        with open(resolved_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=ConfigLoader)

    def resolve_path(self, node: yaml.Node) -> str:
        """Resolve a file path relative to the current config file.

        This is useful for paths that need to be resolved at load time but
        not included as configuration (e.g., model weights, data files, etc.).

        Unlike !include which loads the content, this just resolves the path
        and verifies the file exists.

        Parameters
        ----------
        node : yaml.Node
            YAML node containing the filename

        Returns
        -------
        str
            Resolved absolute path

        Raises
        ------
        ConfigIncludeError
            If file not found

        Examples
        --------
        post:
          flash_match:
            cfg: !path flashmatch/config.yaml  # Resolved relative to this config
        model:
          weights: !path weights/model.ckpt  # Must exist at load time
        """
        filename = self.construct_scalar(cast(yaml.ScalarNode, node))
        resolved_path = resolve_config_path(filename, self._root)
        return resolved_path

    def download(self, node: yaml.Node) -> str:
        """Download a file from URL and return the cached path.

        This is useful for downloading model weights or other large files
        that should not be stored in git but need to be accessible.

        Files are cached in a weights/ directory (or SPINE_CACHE_DIR if set).
        If a file already exists in cache, it is not re-downloaded.

        Parameters
        ----------
        node : yaml.Node
            YAML node containing the URL (string or dict with url/hash)

        Returns
        -------
        str
            Absolute path to cached file

        Raises
        ------
        HTTPError
            If download fails
        ValueError
            If hash validation fails

        Examples
        --------
        model:
          # Simple URL
          weights: !download https://example.com/model.ckpt

          # With hash validation
          weights: !download
            url: https://example.com/model.ckpt
            hash: abc123def456...  # SHA256 hash
        """
        if isinstance(node, yaml.ScalarNode):
            # Simple case: just a URL string
            url = self.construct_scalar(node)
            return download_from_url(url)
        elif isinstance(node, yaml.MappingNode):
            # Complex case: dict with url and optional hash
            data = self.construct_mapping(node)
            url = data.get("url")
            expected_hash = data.get("hash")

            if not url:
                raise ConfigIncludeError(
                    "!download requires either a URL string or a dict with 'url' key"
                )

            return download_from_url(url, expected_hash=expected_hash)
        else:
            raise ConfigIncludeError(
                f"!download expects a string URL or dict, got {type(node)}"
            )


# Register the !include, !path, and !download constructors
ConfigLoader.add_constructor("!include", ConfigLoader.include)
ConfigLoader.add_constructor("!path", ConfigLoader.resolve_path)
ConfigLoader.add_constructor("!download", ConfigLoader.download)


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


def _load_config_recursive(
    cfg_path: str,
    include_stack: Optional[List[str]] = None,
    compatibility_checks: Optional[List[Tuple[Dict, Dict, str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """Recursively load config with cycle detection.

    Parameters
    ----------
    cfg_path : str
        Path to configuration file
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
    """
    cfg_path = os.path.abspath(cfg_path)

    # Cycle detection
    if include_stack is None:
        include_stack = []
    if compatibility_checks is None:
        compatibility_checks = []

    if cfg_path in include_stack:
        cycle = include_stack + [cfg_path]
        raise ConfigCycleError(cycle)

    include_stack = include_stack + [cfg_path]

    root_dir = os.path.dirname(cfg_path)

    # Load YAML
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            main_config = yaml.load(f, Loader=ConfigLoader)
    except FileNotFoundError as exc:
        raise ConfigIncludeError(f"Configuration file not found: {cfg_path}") from exc
    except Exception as exc:
        raise ConfigIncludeError(f"Error loading {cfg_path}: {exc}") from exc

    if main_config is None:
        return {}, {}, [], {}

    # Extract metadata
    metadata = extract_metadata(main_config, cfg_path)
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
        ) = _load_config_recursive(include_path, include_stack, compatibility_checks)

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


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load a SPINE configuration file.

    This is the main entry point for loading configuration files.
    Supports:
    - Hierarchical includes with cycle detection
    - Metadata via __meta__ blocks
    - Override semantics with dot-notation
    - Collection operations (list append/remove, dict key removal)
    - Configurable strict modes (warn/error)

    See module docstring for full configuration language spec.

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
    >>> config = load_config("config.yaml")
    >>> print(config['io']['reader']['batch_size'])
    32
    """
    # Load recursively, accumulating compatibility checks
    compatibility_checks = []
    config, overrides, removals, metadata = _load_config_recursive(
        cfg_path, compatibility_checks=compatibility_checks
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
