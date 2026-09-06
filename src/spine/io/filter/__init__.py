"""Reusable entry-inspection, scan-cache and manifest APIs.

Entry filtering runs before the normal SPINE data-loading lifecycle.  The
package keeps the persistent artifact format independent of any one input
backend while allowing inspectors to perform format-specific measurements.
"""

from .base import EntryInspector, SourceFingerprint, resolve_sources
from .larcv import LArCVEntryInspector
from .manager import (
    build_manifest,
    eligible_entries_from_manifest,
    load_entry_filter,
    load_filter_config,
    scan_sources,
)

__all__ = [
    "EntryInspector",
    "LArCVEntryInspector",
    "SourceFingerprint",
    "build_manifest",
    "eligible_entries_from_manifest",
    "load_entry_filter",
    "load_filter_config",
    "resolve_sources",
    "scan_sources",
]
