"""Configuration metadata extraction and validation for SPINE.

This module provides utilities for handling configuration metadata,
including extraction, validation, and compatibility checking.
"""

import os
import warnings
from typing import Any, Dict, Optional, Tuple

from .api import (
    DEFAULT_KIND,
    DEFAULT_LIST_APPEND,
    META_COMPATIBLE_WITH,
    META_KEY,
    META_KIND,
    META_LIST_APPEND,
    META_STRICT,
    META_VERSION,
    VALID_KINDS,
    VALID_LIST_APPEND_MODES,
    VALID_STRICT_MODES,
)
from .errors import ConfigValidationError


def _compare_versions(actual: str, operator: str, required: str) -> bool:
    """Compare two version strings using the specified operator.

    Only supports YYMMDD format (exactly 6 digits, e.g., "240719").

    Parameters
    ----------
    actual : str
        Actual version from parent config
    operator : str
        Comparison operator: "==", ">=", "<=", ">", "<", "!="
    required : str
        Required version from compatibility constraint

    Returns
    -------
    bool
        True if comparison passes, False otherwise

    Raises
    ------
    ValueError
        If version format is not exactly 6 digits
    """
    # Validate actual version
    if not (actual and actual.isdigit() and len(actual) == 6):
        raise ValueError(
            f"Invalid version format: '{actual}'. Must be exactly 6 digits (YYMMDD format, e.g., '240719')"
        )

    # Validate required version
    if not (required and required.isdigit() and len(required) == 6):
        raise ValueError(
            f"Invalid version format: '{required}'. Must be exactly 6 digits (YYMMDD format, e.g., '240719')"
        )

    # Convert to integers for comparison
    actual_cmp = int(actual)
    required_cmp = int(required)

    # Perform comparison
    if operator == "==":
        return actual_cmp == required_cmp
    elif operator == ">=":
        return actual_cmp >= required_cmp
    elif operator == "<=":
        return actual_cmp <= required_cmp
    elif operator == ">":
        return actual_cmp > required_cmp
    elif operator == "<":
        return actual_cmp < required_cmp
    elif operator == "!=":
        return actual_cmp != required_cmp
    else:
        warnings.warn(
            f"Unknown version operator '{operator}', treating as '=='", stacklevel=3
        )
        return actual_cmp == required_cmp


def _parse_version_constraint(constraint: str) -> Tuple[str, str]:
    """Parse a version constraint into operator and version.

    Parameters
    ----------
    constraint : str
        Version constraint like ">=240719", "==1.0.0", or "240719"

    Returns
    -------
    Tuple[str, str]
        (operator, version) tuple. If no operator, returns ("==", version)
    """
    constraint = constraint.strip()

    # Check for operators (order matters - check >= before >)
    for op in [">=", "<=", "==", "!=", ">", "<"]:
        if constraint.startswith(op):
            return op, constraint[len(op) :].strip()

    # No operator means exact match
    return "==", constraint


def extract_metadata(
    config_dict: Dict[str, Any], cfg_path: Optional[str] = None
) -> Dict[str, Any]:
    """Extract and validate __meta__ block from config.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary
    cfg_path : str, optional
        Path to config file (for metadata tracking)

    Returns
    -------
    Dict[str, Any]
        Metadata dict with defaults filled in
    """
    meta = config_dict.get(META_KEY, {})

    # Validate kind
    kind = meta.get(META_KIND, DEFAULT_KIND)
    if kind not in VALID_KINDS:
        warnings.warn(
            f"Invalid __meta__.kind: '{kind}', must be one of {VALID_KINDS}. Using '{DEFAULT_KIND}'"
        )
        kind = DEFAULT_KIND

    # Validate strict mode
    strict = meta.get(META_STRICT)
    if strict is None:
        # Default: warn for mods, error for bundles
        strict = "warn" if kind == "mod" else "error"
    elif strict not in VALID_STRICT_MODES:
        warnings.warn(
            f"Invalid __meta__.strict: '{strict}', must be one of {VALID_STRICT_MODES}. Using default"
        )
        strict = "warn" if kind == "mod" else "error"

    # Validate list_append mode
    list_append = meta.get(META_LIST_APPEND, DEFAULT_LIST_APPEND)
    if list_append not in VALID_LIST_APPEND_MODES:
        warnings.warn(
            f"Invalid __meta__.list_append: '{list_append}', must be one of {VALID_LIST_APPEND_MODES}. Using '{DEFAULT_LIST_APPEND}'"
        )
        list_append = DEFAULT_LIST_APPEND

    result = {
        META_KIND: kind,
        META_STRICT: strict,
        META_LIST_APPEND: list_append,
        **meta,  # Include other metadata fields as-is
    }

    # Add file path for compatibility checking
    if cfg_path:
        result["_file_path"] = cfg_path

    return result


def check_compatibility(
    parent_meta: Dict[str, Any], included_meta: Dict[str, Any], included_path: str
) -> None:
    """Check if included config is compatible with parent.

    Validates that an included configuration meets the parent's
    compatibility requirements. Supports three formats:

    1. String: compatible_with: "base_name"
    2. List: compatible_with: [name1, name2]
    3. Dict with version constraints: compatible_with: {base: ">=240719", io: "==240812"}

    Parameters
    ----------
    parent_meta : Dict[str, Any]
        Parent config metadata (must include 'components' dict for version checking)
    included_meta : Dict[str, Any]
        Included config metadata
    included_path : str
        Path to included file (for error messages)

    Raises
    ------
    ConfigValidationError
        If included config is incompatible with parent requirements
    UserWarning
        If potential compatibility issues detected
    """
    # Check if included config declares compatibility requirements
    compatible_with = included_meta.get(META_COMPATIBLE_WITH)
    if not compatible_with:
        return

    # Handle dictionary format with version constraints
    # Example: {base: "==240719", io: ">=240812"}
    if isinstance(compatible_with, dict):
        # Get parent's component versions
        # Components can come from two places:
        # 1. Accumulated from previous includes (built up during loading)
        # 2. Declared upfront in parent's __meta__.components (for top-level configs)
        parent_components = parent_meta.get("components", {})
        parent_version = parent_meta.get(META_VERSION)

        # Track failures for detailed error message
        failures = []

        for component, constraint in compatible_with.items():
            # Get actual version from parent's components dict
            actual_version = parent_components.get(component)

            # If component not found in components dict, try parent's own version
            # (for backwards compatibility with simpler configs without component tracking)
            if actual_version is None:
                actual_version = parent_version

            # Parse the constraint
            operator, required_version = _parse_version_constraint(constraint)

            # Perform version comparison
            if actual_version is None:
                failures.append(
                    f"{component}: parent has no version (required {operator}{required_version})"
                )
            elif not _compare_versions(actual_version, operator, required_version):
                failures.append(
                    f"{component}: {actual_version} does not satisfy {operator}{required_version}"
                )

        # If any constraints failed, raise error
        if failures:
            file_name = os.path.basename(included_path)
            parent_file = os.path.basename(parent_meta.get("_file_path", "<unknown>"))
            failure_details = "; ".join(failures)

            raise ConfigValidationError(
                f"Compatibility error: '{file_name}' has incompatible version requirements.\n"
                f"  Parent: '{parent_file}'\n"
                f"  Failed constraints: {failure_details}"
            )

        # All constraints passed
        return

    # Handle string/list format (legacy simple matching)
    # Get parent's kind and extends field
    parent_kind = parent_meta.get(META_KIND, DEFAULT_KIND)
    parent_extends = parent_meta.get("extends")
    parent_version = parent_meta.get(META_VERSION)

    # Get included's extends field
    included_extends = included_meta.get("extends")

    # Build list of parent identifiers to check against
    parent_identifiers = {parent_kind}
    if parent_extends:
        if isinstance(parent_extends, list):
            parent_identifiers.update(parent_extends)
        else:
            parent_identifiers.add(parent_extends)

    # Add version-qualified identifiers if parent has version
    if parent_version:
        versioned_ids = {f"{pid}_{parent_version}" for pid in parent_identifiers}
        parent_identifiers.update(versioned_ids)

    # Normalize compatible_with to list
    if isinstance(compatible_with, str):
        compatible_with = [compatible_with]

    # Check if any parent identifier matches compatible_with
    if not any(pid in compatible_with for pid in parent_identifiers):
        file_name = os.path.basename(included_path)
        parent_file = os.path.basename(parent_meta.get("_file_path", "<unknown>"))

        raise ConfigValidationError(
            f"Compatibility error: '{file_name}' declares compatible_with={compatible_with}, "
            f"but parent '{parent_file}' has kind='{parent_kind}', extends={parent_extends}, version={parent_version}. "
            f"The included config cannot be used in this context."
        )

    # Check for conflicting modifiers (same 'extends' value)
    if included_extends and parent_extends:
        # Normalize to sets for comparison
        included_set = (
            {included_extends}
            if isinstance(included_extends, str)
            else set(included_extends)
        )
        parent_set = (
            {parent_extends} if isinstance(parent_extends, str) else set(parent_extends)
        )

        # Check for conflicts (same modifier type)
        conflicts = included_set & parent_set
        if conflicts:
            file_name = os.path.basename(included_path)
            parent_file = os.path.basename(parent_meta.get("_file_path", "<unknown>"))
            warnings.warn(
                f"Potential modifier conflict: Both '{parent_file}' and '{file_name}' "
                f"extend {conflicts}. This may cause unexpected behavior.",
                stacklevel=4,
            )
