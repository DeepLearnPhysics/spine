"""API version and metadata constants for SPINE configuration loading.

This module defines the configuration API version and metadata schema.
"""

# Current API version
API_VERSION = "1.0"

# Metadata schema
META_KEY = "__meta__"

# Metadata fields
META_VERSION = "version"
META_DATE = "date"
META_DESCRIPTION = "description"
META_COMPATIBLE_WITH = "compatible_with"
META_TAGS = "tags"
META_KIND = "kind"  # "bundle" or "mod"
META_STRICT = "strict"  # "warn" or "error"
META_LIST_APPEND = "list_append"  # "append" or "unique"

# Default values
DEFAULT_KIND = "bundle"
DEFAULT_STRICT = "warn"  # warn for mods, error for bundles (see loader)
DEFAULT_LIST_APPEND = "append"  # current behavior

# Valid values
VALID_KINDS = {"bundle", "mod"}
VALID_STRICT_MODES = {"warn", "error"}
VALID_LIST_APPEND_MODES = {"append", "unique"}
