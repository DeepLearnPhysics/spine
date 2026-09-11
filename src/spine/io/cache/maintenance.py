"""Maintenance operations for manifest-backed cache repositories."""

from __future__ import annotations

import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .manifest import CacheStage

if TYPE_CHECKING:  # pragma: no cover
    from .repository import CacheRepository

__all__ = ["CacheGarbageCollection", "collect_garbage", "remove_retired_stages"]


@dataclass(frozen=True)
class CacheGarbageCollection:
    """Summary of one cache garbage-collection pass.

    Attributes
    ----------
    shard_generations : tuple[str, ...]
        Unreferenced immutable generation directories selected for removal.
    pending_transactions : tuple[str, ...]
        Abandoned transaction directories selected for removal.
    total_bytes : int
        Combined size of the selected files before removal.
    dry_run : bool
        Whether the pass only reported candidates without modifying them.
    """

    shard_generations: tuple[str, ...]
    pending_transactions: tuple[str, ...]
    total_bytes: int
    dry_run: bool


def remove_retired_stages(
    repository: CacheRepository,
    retired: Iterable[CacheStage],
    live: Iterable[CacheStage],
) -> None:
    """Remove exact shard files retired by a successful manifest update.

    The caller must publish the replacement manifest before invoking this
    function. Only paths named by the retired records are considered, and a
    defensive live-reference check prevents removal if records share a shard.

    Parameters
    ----------
    repository : CacheRepository
        Repository whose published generation was replaced.
    retired, live : iterable[CacheStage]
        Stage records removed from and retained by the new manifest.

    Notes
    -----
    Cleanup failures do not roll back an already published manifest. They are
    reported as warnings and the leftover files can later be removed by the
    explicit garbage collector.
    """
    live_paths = {
        _shard_path(repository, relative)
        for stage in live
        for relative in stage.shards.values()
    }
    retired_paths = {
        _shard_path(repository, relative)
        for stage in retired
        for relative in stage.shards.values()
    }
    generation_dirs: set[Path] = set()
    failures = []

    # Delete only files named by the retired manifest snapshot. A repository
    # scan here could race with another writer preparing unpublished shards.
    for path in sorted(retired_paths.difference(live_paths)):
        generation_dirs.add(path.parent)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failures.append(f"{path}: {exc}")

    for directory in sorted(
        generation_dirs, key=lambda path: len(path.parts), reverse=True
    ):
        _remove_empty_parents(directory, repository.shard_dir)

    if len(failures) > 0:
        warnings.warn(
            "Cache replacement was published, but retired shard cleanup was "
            "incomplete; run `spine-cache gc` later. " + "; ".join(failures),
            RuntimeWarning,
            stacklevel=2,
        )


def collect_garbage(
    repository: CacheRepository,
    *,
    dry_run: bool = False,
    min_age_seconds: float = 86400.0,
) -> CacheGarbageCollection:
    """Remove stale files which are unreachable from the current manifest.

    Parameters
    ----------
    repository : CacheRepository
        Repository to inspect and, unless ``dry_run`` is set, clean.
    dry_run : bool, default False
        Report eligible paths and bytes without deleting anything.
    min_age_seconds : float, default 86400
        Minimum time since the latest transaction activity or file update.
        The one-day default prevents cleanup from racing normally active jobs.

    Returns
    -------
    CacheGarbageCollection
        Selected generation directories, abandoned transactions, and their
        combined size.

    Raises
    ------
    ValueError
        If ``min_age_seconds`` is negative.

    Notes
    -----
    The manifest lock stabilizes the live reference set during collection.
    Active transactions publish a heartbeat in ``pending``; their matching
    shard generation is protected even after files begin moving into place.
    """
    if min_age_seconds < 0:
        raise ValueError("Cache garbage-collection minimum age cannot be negative.")

    cutoff = time.time() - min_age_seconds
    with repository._manifest_lock():
        manifest = repository.load()
        live_paths = {
            _shard_path(repository, relative)
            for stage in (*manifest.stages.values(), *manifest.replacements.values())
            for relative in stage.shards.values()
        }

        pending = _generation_directories(repository.pending_dir)
        fresh_pending = {
            directory.name for directory in pending if _latest_mtime(directory) > cutoff
        }
        stale_pending = tuple(
            directory for directory in pending if _latest_mtime(directory) <= cutoff
        )

        stale_shards = []
        for directory in _generation_directories(repository.shard_dir, nested=True):
            files = {path.resolve() for path in directory.rglob("*") if path.is_file()}
            if files.intersection(live_paths):
                continue
            if directory.name in fresh_pending:
                continue
            if _latest_mtime(directory) <= cutoff:
                stale_shards.append(directory)

        stale_shard_tuple = tuple(stale_shards)
        total_bytes = sum(
            _directory_size(path) for path in (*stale_shard_tuple, *stale_pending)
        )

        if not dry_run:
            for path in (*stale_shard_tuple, *stale_pending):
                shutil.rmtree(path)
            for path in stale_shard_tuple:
                _remove_empty_parents(path.parent, repository.shard_dir)

    return CacheGarbageCollection(
        shard_generations=tuple(
            str(path.relative_to(repository.path)) for path in stale_shard_tuple
        ),
        pending_transactions=tuple(
            str(path.relative_to(repository.path)) for path in stale_pending
        ),
        total_bytes=total_bytes,
        dry_run=dry_run,
    )


def _shard_path(repository: CacheRepository, relative_path: str) -> Path:
    """Resolve a manifest shard path without requiring it to still exist."""
    path = (repository.path / relative_path).resolve()
    if repository.shard_dir not in path.parents:
        raise ValueError(f"Cache shard path escapes the shard directory: {path}.")
    return path


def _generation_directories(root: Path, nested: bool = False) -> tuple[Path, ...]:
    """List UUID generation directories in a cache-maintenance root."""
    if not root.is_dir():
        return ()
    if not nested:
        return tuple(path for path in root.iterdir() if path.is_dir())
    return tuple(
        generation
        for stage in root.iterdir()
        if stage.is_dir()
        for generation in stage.iterdir()
        if generation.is_dir()
    )


def _latest_mtime(path: Path) -> float:
    """Return the latest modification time within a transaction directory."""
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        latest = max(latest, child.stat().st_mtime)
    return latest


def _directory_size(path: Path) -> int:
    """Measure the regular-file payload below a generation directory."""
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _remove_empty_parents(path: Path, stop: Path) -> None:
    """Remove empty generation/stage directories without crossing ``stop``."""
    current = path
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
