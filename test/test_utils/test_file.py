"""Tests for collaboration-oriented filesystem permission helpers."""

import os
import stat

from spine.utils.file import (
    SHARED_DIRECTORY_MODE,
    SHARED_FILE_MODE,
    make_shared_directory,
    set_shared_file_permissions,
    set_shared_tree_permissions,
)


def mode(path):
    """Return the portable permission bits for one filesystem path."""
    return stat.S_IMODE(path.stat().st_mode)


def test_shared_permissions_override_restrictive_umask(tmp_path):
    """Generated trees should remain collaboration-readable under umask 077."""
    old_umask = os.umask(0o077)
    try:
        root = tmp_path / "shared" / "nested"
        make_shared_directory(root, parents=True)
        make_shared_directory(root, exist_ok=True)

        direct = root / "direct.dat"
        direct.write_text("data", encoding="utf-8")
        set_shared_file_permissions(direct)

        child = root / "child"
        child.mkdir()
        nested = child / "nested.dat"
        nested.write_text("data", encoding="utf-8")
        set_shared_tree_permissions(root)
    finally:
        os.umask(old_umask)

    assert mode(tmp_path / "shared") == SHARED_DIRECTORY_MODE
    assert mode(root) == SHARED_DIRECTORY_MODE
    assert mode(child) == SHARED_DIRECTORY_MODE
    assert mode(direct) == SHARED_FILE_MODE
    assert mode(nested) == SHARED_FILE_MODE
