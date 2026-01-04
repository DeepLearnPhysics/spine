"""Typed exceptions for SPINE configuration loading.

This module defines specific exception types for different kinds of
configuration errors, making it easier to handle and debug issues.
"""

from typing import List


class ConfigError(Exception):
    """Base exception for all configuration errors."""


class ConfigIncludeError(ConfigError):
    """Raised when an included file cannot be found or loaded."""


class ConfigCycleError(ConfigError):
    """Raised when a circular include dependency is detected."""

    def __init__(self, cycle_path: List[str]):
        """Initialize with the cycle path.

        Parameters
        ----------
        cycle_path : List[str]
            List of file paths showing the include cycle
        """
        self.cycle_path = cycle_path
        cycle_str = " -> ".join(cycle_path)
        super().__init__(f"Circular include detected: {cycle_str}")


class ConfigPathError(ConfigError):
    """Raised when a configuration path cannot be resolved or does not exist."""


class ConfigTypeError(ConfigError):
    """Raised when a configuration operation is applied to the wrong type."""


class ConfigOperationError(ConfigError):
    """Raised when a configuration operation (e.g., collection op) fails."""


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails (e.g., compatibility checks)."""
