"""Transactional publication of one immutable cache-stage generation."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from spine.utils.file import make_shared_directory, set_shared_file_permissions

from .manifest import CacheSource, CacheStage
from .repository import CacheRepository

__all__ = ["CacheTransaction"]


class CacheTransaction:
    """Own the private write area and publication of one cache stage.

    A transaction captures the manifest generation visible at construction,
    writes new shards under a unique pending directory, and publishes them only
    after the physical backend has finalized and validated every shard.
    """

    def __init__(
        self, repository: CacheRepository, stage: str, overwrite: bool = False
    ) -> None:
        """Create a private pending generation against the current manifest.

        Parameters
        ----------
        repository : CacheRepository
            Repository which will own the published generation.
        stage : str
            Logical processing-stage name.
        overwrite : bool, default False
            Permit replacement of an existing stage generation.

        Raises
        ------
        ValueError
            If ``stage`` is not a safe, nonempty path component.
        """
        if not stage or Path(stage).name != stage or stage in (".", ".."):
            raise ValueError("Cache stage names must be nonempty path components.")
        self.repository = repository
        self.stage = stage
        self.overwrite = overwrite
        self.snapshot = repository.load()
        self.base_generation = self.snapshot.generation
        self.generation = uuid.uuid4().hex
        self.pending_path = repository.pending_dir / self.generation
        make_shared_directory(self.pending_path, parents=True)
        self.published = False

    def publish(
        self,
        pending_by_source: dict[str, Path],
        sources: tuple[CacheSource, ...],
        products: tuple[str, ...],
    ) -> None:
        """Move validated shards into place and publish their manifest record.

        Parameters
        ----------
        pending_by_source : dict[str, pathlib.Path]
            Completed pending shard for each stable source identifier.
        sources : tuple[CacheSource, ...]
            Ordered source records represented by the new stage.
        products : tuple[str, ...]
            Public product schema shared by all stage shards.

        Notes
        -----
        Shards are renamed into their immutable generation directory before
        the manifest commit. Until that final commit succeeds, no reader can
        discover the new generation.
        """
        # Each generation receives a fresh immutable directory. It remains
        # unreachable to readers until the manifest publication below.
        generation_dir = self.repository.shard_dir / self.stage / self.generation
        make_shared_directory(generation_dir, parents=True)

        # Rename on the repository filesystem is atomic and avoids a second
        # copy of the newly written stage before its short manifest commit.
        shards = {}
        for source in sources:
            destination = generation_dir / f"{source.id}.h5"
            pending = pending_by_source[source.id]
            set_shared_file_permissions(pending)
            os.replace(pending, destination)
            shards[source.id] = str(destination.relative_to(self.repository.path))

        stage_record = CacheStage(
            generation=self.generation,
            products=products,
            shards=shards,
        )
        self.repository.publish_stage(
            self.stage,
            stage_record,
            sources,
            self.base_generation,
            overwrite=self.overwrite,
        )
        self.published = True
        self.cleanup()

    def cleanup(self) -> None:
        """Discard this transaction's unpublished pending files.

        Cleanup is scoped to the transaction's UUID-named directory and is
        safe to call after successful publication or during error handling.
        Published immutable shards are never removed here.
        """
        if self.pending_path.exists():
            shutil.rmtree(self.pending_path)
