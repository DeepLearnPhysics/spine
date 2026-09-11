"""Tests for the standalone cache-maintenance command."""

import os
import time

from spine.bin.cache import cli
from spine.io.cache import CacheRepository


def test_cache_gc_cli_reports_dry_run(tmp_path, capsys):
    """The GC command should expose a non-mutating inspection mode."""
    path = tmp_path / "empty.spine-cache"
    repository = CacheRepository(str(path), create=True)
    abandoned = repository.pending_dir / "abandoned"
    abandoned.mkdir()
    marker = abandoned / ".activity"
    marker.write_bytes(b"x")
    old_time = time.time() - 10
    os.utime(marker, (old_time, old_time))
    os.utime(abandoned, (old_time, old_time))

    assert cli(["gc", str(path), "--dry-run", "--min-age", "1"]) == 0
    output = capsys.readouterr().out
    assert "Would remove 1 cache generations (1 bytes)." in output
    assert "pending/abandoned" in output
