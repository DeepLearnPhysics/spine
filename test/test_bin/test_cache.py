"""Tests for the standalone cache-maintenance command."""

import os
import time
from unittest.mock import patch

from spine.bin.cache import cli
from spine.io.cache import CacheRepository


def test_cache_begin_cli_registers_given_and_generated_ids(tmp_path, capsys):
    """The begin command should create a repository and print its fence ID."""
    path = tmp_path / "train.spine-cache"
    assert cli(["begin", str(path), "stage", "--publication-id", "submission-1"]) == 0
    assert capsys.readouterr().out == "submission-1\n"
    assert CacheRepository(str(path)).load().publications == {"stage": "submission-1"}

    with patch("spine.bin.cache.uuid.uuid4") as make_uuid:
        make_uuid.return_value.hex = "generated-id"
        assert cli(["begin", str(path), "stage"]) == 0
    assert capsys.readouterr().out == "generated-id\n"
    assert CacheRepository(str(path)).load().publications == {"stage": "generated-id"}


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
