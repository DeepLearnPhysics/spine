"""Filesystem lifecycle for manifest-backed SPINE cache repositories."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from .manifest import CacheManifest, CacheSource, CacheStage

__all__ = ["CacheRepository"]


class CacheRepository:
    """Manage the filesystem layout of a logical cache repository.

    A repository contains one atomically replaced manifest, immutable HDF5
    shards grouped by stage generation, and private pending directories used
    by active writers. Readers retain a consistent manifest snapshot while a
    later generation is prepared and published.
    """

    manifest_name = "manifest.json"

    def __init__(self, path: str, create: bool = False) -> None:
        """Initialize a repository path.

        Parameters
        ----------
        path : str
            Cache repository directory, conventionally ending in
            ``.spine-cache``.
        create : bool, default False
            Create an empty repository when it does not yet exist.

        Raises
        ------
        ValueError
            If creation targets a nonempty directory that is not already a
            cache repository.
        FileNotFoundError
            If ``create=False`` and no published manifest exists.
        """
        self.path = Path(path).expanduser().resolve()
        self.manifest_path = self.path / self.manifest_name
        self.shard_dir = self.path / "shards"
        self.pending_dir = self.path / "pending"
        self.lock_path = self.path / ".manifest.lock"

        if create:
            if (
                self.path.exists()
                and not self.manifest_path.exists()
                and any(self.path.iterdir())
            ):
                raise ValueError(
                    f"Cannot initialize cache repository in nonempty directory "
                    f"'{self.path}'."
                )
            self.path.mkdir(parents=True, exist_ok=True)
            self.shard_dir.mkdir(exist_ok=True)
            self.pending_dir.mkdir(exist_ok=True)
            with self._manifest_lock():
                if not self.manifest_path.exists():
                    self._write_manifest(CacheManifest())
        elif not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Cache repository '{self.path}' has no {self.manifest_name}."
            )

    def load(self) -> CacheManifest:
        """Read and validate the currently published manifest snapshot.

        Returns
        -------
        CacheManifest
            Immutable description of the visible source and stage set.

        Raises
        ------
        TypeError
            If the JSON document does not contain an object at its root.
        ValueError
            If decoded manifest contents violate the cache schema.
        """
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError("Cache manifest root must be a JSON object.")
        return CacheManifest.from_dict(data)

    def resolve_shard(self, relative_path: str) -> str:
        """Resolve and validate a shard path stored in the manifest.

        Parameters
        ----------
        relative_path : str
            Repository-relative path recorded for one published shard.

        Returns
        -------
        str
            Absolute path to the existing shard.

        Raises
        ------
        ValueError
            If the resolved path escapes the repository directory.
        FileNotFoundError
            If the published shard no longer exists.
        """
        path = (self.path / relative_path).resolve()
        if self.path not in path.parents:
            raise ValueError(f"Cache shard path escapes the repository: {path}.")
        if not path.is_file():
            raise FileNotFoundError(f"Published cache shard does not exist: {path}.")
        return str(path)

    def publish_stage(
        self,
        stage_name: str,
        stage: CacheStage,
        sources: tuple[CacheSource, ...],
        base_generation: int,
        overwrite: bool = False,
    ) -> CacheManifest:
        """Atomically add or replace one completed stage generation.

        A lock serializes writers only for the short manifest commit. Shards
        are fully written and validated before this method is called, so active
        readers continue using their previous immutable snapshot throughout.

        Parameters
        ----------
        stage_name : str
            Logical stage to add or replace.
        stage : CacheStage
            Completed immutable generation being published.
        sources : tuple[CacheSource, ...]
            Ordered source set represented by the generation.
        base_generation : int
            Manifest generation observed when the write transaction began.
        overwrite : bool, default False
            Permit replacement of an existing logical stage.

        Returns
        -------
        CacheManifest
            Newly published manifest snapshot.

        Raises
        ------
        RuntimeError
            If another writer published first, or the stage already exists
            without explicit replacement permission.
        ValueError
            If the stage does not use the repository's established source set.
        """
        with self._manifest_lock():
            current = self.load()
            if current.generation != base_generation:
                raise RuntimeError(
                    "Cache manifest changed while this stage was being written; "
                    "restart the stage against the latest cache snapshot."
                )
            if stage_name in current.stages and not overwrite:
                raise RuntimeError(
                    f"Cache stage '{stage_name}' is already published. Set "
                    "overwrite_stage=True to replace it."
                )
            if current.sources and current.sources != sources:
                raise ValueError(
                    "Published cache stages must use the same ordered source set."
                )

            # Construct a new immutable snapshot; existing stage generations
            # remain untouched and available to readers holding older state.
            stages = dict(current.stages)
            stages[stage_name] = stage
            updated = replace(
                current,
                generation=current.generation + 1,
                sources=current.sources or sources,
                stages=stages,
            )
            self._write_manifest(updated)
            return updated

    @contextmanager
    def _manifest_lock(self) -> Iterator[None]:
        """Hold an advisory exclusive lock around one manifest update.

        Yields
        ------
        None
            Control while the repository lock is held by this process.
        """
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _write_manifest(self, manifest: CacheManifest) -> None:
        """Durably replace the manifest with a complete JSON snapshot.

        Parameters
        ----------
        manifest : CacheManifest
            Fully validated snapshot to publish.

        Notes
        -----
        The temporary file is flushed and synchronized before an atomic rename,
        so readers never observe a partially serialized manifest.
        """
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)
