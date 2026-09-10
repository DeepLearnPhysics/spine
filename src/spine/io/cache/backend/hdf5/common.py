"""Shared helpers for the private HDF5 cache backend."""

import hashlib
from typing import Any

__all__ = ["source_id"]


def source_id(source_info: dict[str, Any]) -> str:
    """Return a stable compact identifier for one source-file identity.

    Parameters
    ----------
    source_info : dict
        Mapping containing file name, size and modification timestamp.

    Returns
    -------
    str
        First 20 hexadecimal characters of the source identity hash.

    Notes
    -----
    All three provenance fields participate in the digest so files with the
    same basename remain distinct and a modified source cannot silently reuse
    an older shard.
    """
    identity = "\0".join(
        (
            str(source_info["file_name"]),
            str(int(source_info["file_size"])),
            str(int(source_info["file_mtime_ns"])),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
