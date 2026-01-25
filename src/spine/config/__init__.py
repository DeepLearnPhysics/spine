"""SPINE configuration loading system.

This package provides a sophisticated configuration loading system with:
- Hierarchical file includes with cycle detection
- Metadata support for versioning and configuration
- Override semantics with dot-notation
- Collection operations (list/dict)
- Configurable strict modes

Main Entry Point
----------------
load_config : Load a SPINE configuration file

See loader module docstring for full configuration language specification.
"""

from .api import API_VERSION
from .download import download_from_url, get_cache_dir
from .errors import (
    ConfigCycleError,
    ConfigError,
    ConfigIncludeError,
    ConfigOperationError,
    ConfigPathError,
    ConfigTypeError,
    ConfigValidationError,
)
from .loader import load_config

__version__ = API_VERSION

__all__ = [
    "load_config",
    "download_from_url",
    "get_cache_dir",
    "ConfigError",
    "ConfigIncludeError",
    "ConfigCycleError",
    "ConfigPathError",
    "ConfigTypeError",
    "ConfigOperationError",
    "ConfigValidationError",
]
