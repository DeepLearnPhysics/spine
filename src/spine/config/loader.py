"""YAML loader with SPINE-specific features.

This module provides the ConfigLoader class which extends yaml.SafeLoader
with support for:
- !include tag for inline file includes
- !path tag for path resolution
- !download tag for downloading files from URLs
"""

import os
from typing import Any, List, Optional, TextIO, Union, cast

import yaml

from .download import download_from_url
from .errors import ConfigIncludeError

__all__ = ["ConfigLoader", "resolve_config_path"]


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

    def __init__(
        self, stream: Union[TextIO, str], root_dir: Optional[str] = None
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        stream : Union[TextIO, str]
            File stream (from `open()`) or string content
        root_dir : Optional[str]
            Root directory for resolving relative paths.
            If None and stream is a file, uses the file's directory.
        """
        if isinstance(stream, str):
            # String input
            self._root = root_dir if root_dir is not None else os.getcwd()
        else:
            # File stream
            if root_dir is not None:
                self._root = root_dir
            else:
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
            # Create a loader class with the resolved directory as root_dir
            resolved_dir = os.path.dirname(resolved_path)

            class NestedConfigLoader(ConfigLoader):
                def __init__(self, stream):
                    super().__init__(stream, resolved_dir)

            return yaml.load(f, Loader=NestedConfigLoader)

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
