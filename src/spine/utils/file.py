"""Shared filesystem-permission helpers for collaboration artifacts."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "SHARED_DIRECTORY_MODE",
    "SHARED_FILE_MODE",
    "make_shared_directory",
    "set_shared_file_permissions",
    "set_shared_tree_permissions",
]

SHARED_FILE_MODE = 0o664
SHARED_DIRECTORY_MODE = 0o2775


def set_shared_file_permissions(path: str | os.PathLike[str]) -> None:
    """Make a generated regular file reusable by collaborating accounts.

    Parameters
    ----------
    path : path-like
        Existing output artifact whose permissions should be normalized.

    Notes
    -----
    SPINE output artifacts are deliberately readable by everyone and writable
    by their owner and group. Applying the mode explicitly also repairs the
    secure ``0600`` mode imposed by :mod:`tempfile` before atomic publication.
    """
    Path(path).chmod(SHARED_FILE_MODE)


def make_shared_directory(
    path: str | os.PathLike[str],
    *,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    """Create collaboration-friendly output directories.

    Parameters
    ----------
    path : path-like
        Directory to create.
    parents : bool, default False
        Create missing parent directories as needed.
    exist_ok : bool, default False
        Do not fail when the target directory already exists.

    Notes
    -----
    Only directories created by this call are modified. Existing ancestors
    retain their user-managed permissions. The set-group-ID bit makes files
    created below a shared directory inherit its collaboration group.
    """
    directory = Path(path)
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent

    directory.mkdir(parents=parents, exist_ok=exist_ok)

    # Normalize every directory created by a recursive mkdir, from the nearest
    # existing ancestor down to the requested output directory.
    for created in reversed(missing):
        created.chmod(SHARED_DIRECTORY_MODE)


def set_shared_tree_permissions(path: str | os.PathLike[str]) -> None:
    """Normalize permissions throughout a generated output directory.

    Parameters
    ----------
    path : path-like
        Root directory containing files produced by an external writer.

    Notes
    -----
    This is intended for dedicated SPINE output trees such as TensorBoard
    logs, where the third-party writer controls file creation internally.
    """
    root = Path(path)
    root.chmod(SHARED_DIRECTORY_MODE)
    for item in root.rglob("*"):
        if item.is_dir():
            item.chmod(SHARED_DIRECTORY_MODE)
        else:
            set_shared_file_permissions(item)
