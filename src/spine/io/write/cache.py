"""Writer for manifest-backed, immutable-shard SPINE cache repositories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import CacheRepository, CacheSource, CacheTransaction
from ..cache.backend.hdf5.reader import inspect_stage_shard, read_source_entry_index
from ..cache.backend.hdf5.writer import HDF5ShardWriter

__all__ = ["CacheWriter"]


class CacheWriter:
    """Publish one stage as immutable HDF5 shards behind an atomic manifest.

    Each run writes only its new stage into a private pending directory. A
    successful finalization moves those files into immutable generation paths
    and atomically publishes a new repository manifest.
    """

    name = "cache"

    def __init__(
        self,
        path: str | None = None,
        stage: str | None = None,
        keys: list[str] | None = None,
        skip_keys: list[str] | None = None,
        lite: bool = False,
        keep_open: bool = True,
        flush_frequency: int | None = None,
        overwrite_stage: bool = False,
        parallel: bool = False,
        dependencies: dict[str, str] | None = None,
        expected_sources: int | None = None,
        publication_id: str | None = None,
        prefix: str | list[str] | None = None,
        split: bool = False,
    ) -> None:
        """Initialize a transactional cache-stage writer.

        Parameters
        ----------
        path : str, optional
            Logical cache repository directory. This field is required by the
            cache writer but remains optional in the signature for compatibility
            with the generic writer factory.
        stage : str, optional
            Name under which this stage generation will be published. This is
            likewise required for cache output.
        keys, skip_keys, lite, keep_open, flush_frequency : optional
            Product projection and physical HDF5 controls forwarded to the V2
            shard writer.
        overwrite_stage : bool, default False
            Permit the new generation to replace an existing named stage. In
            parallel mode, contributions remain hidden until the complete
            replacement is ready for one atomic reader-visible cutover.
        parallel : bool, default False
            Publish this transaction as one disjoint source contribution. The
            stage becomes readable only after it covers the repository roster.
            When replacing a complete stage, readers continue seeing the old
            generation while this roster is assembled.
        dependencies : dict[str, str], optional
            Upstream stage generations consumed by this stage. Normally filled
            automatically by :class:`~spine.io.manager.IOManager`.
        expected_sources : int, optional
            Total number of source shards expected across all parallel tasks.
            Required when ``parallel=True`` so a failed task cannot leave a
            partial first stage looking complete.
        publication_id : str, optional
            Shared identity for one parallel submission attempt. If omitted,
            read from ``SPINE_CACHE_PUBLICATION_ID``. Normal production
            launchers should provide the environment variable rather than
            placing this orchestration detail in experiment YAML.
        prefix, split : optional
            Generic writer-factory arguments. Cache routing is always by source
            identity, so these values do not affect the repository layout.

        Raises
        ------
        ValueError
            If either the repository path or logical stage name is omitted.
        """
        del prefix, split
        if path is None:
            raise ValueError("CacheWriter requires a repository `path`.")
        if stage is None:
            raise ValueError("CacheWriter requires a `stage` name.")
        if parallel and (expected_sources is None or expected_sources < 1):
            raise ValueError(
                "Parallel CacheWriter requires a positive `expected_sources`."
            )
        if parallel:
            if publication_id is None:
                publication_id = os.environ.get("SPINE_CACHE_PUBLICATION_ID")
            if not publication_id:
                raise ValueError(
                    "Parallel CacheWriter requires a shared `publication_id` or "
                    "SPINE_CACHE_PUBLICATION_ID environment variable."
                )
        elif publication_id is not None:
            raise ValueError("CacheWriter `publication_id` requires `parallel=True`.")

        self.repository = CacheRepository(path, create=True)
        self.stage = stage
        self.overwrite_stage = overwrite_stage
        self.transaction = CacheTransaction(
            self.repository,
            stage,
            overwrite=overwrite_stage,
            parallel=parallel,
            dependencies=dependencies,
            expected_sources=expected_sources,
            publication_id=publication_id,
        )

        # The private HDF5 backend owns schema discovery and product
        # serialization. The repository layer only publishes immutable shards.
        self._writer = HDF5ShardWriter(
            directory=str(self.transaction.pending_path),
            prefix=["cache"],
            stage=stage,
            keys=keys,
            skip_keys=skip_keys,
            lite=lite,
            keep_open=keep_open,
            flush_frequency=flush_frequency,
            source_id_names=True,
        )

    def __call__(self, data: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
        """Append one batch to the unpublished stage generation.

        Parameters
        ----------
        data : dict[str, Any]
            Scalar or batched SPINE products with source-file provenance.
        cfg : dict[str, Any], optional
            Complete run configuration persisted with the cache stage.
        """
        self.transaction.touch()
        self._writer(data, cfg)
        self.transaction.touch()

    def finalize(self) -> None:
        """Validate and atomically publish the completed stage generation.

        Finalization first marks every pending HDF5 shard complete, verifies a
        common product schema and source-entry axis, then hands the immutable
        files to the repository transaction for manifest publication. Calling
        this method again after successful publication has no effect.

        Raises
        ------
        RuntimeError
            If no shards were written or publication loses a concurrent
            manifest-generation race.
        ValueError
            If shards disagree on schema, source identity, or compact-to-source
            entry alignment.
        """
        if self.transaction.published:
            return

        self.transaction.touch()
        self._writer.finalize()
        self._writer.close()
        pending_files = sorted(self.transaction.pending_path.glob("*.h5"))
        if len(pending_files) == 0:
            raise RuntimeError("Cannot publish a cache stage which wrote no shards.")

        # Inspect every closed shard before any file becomes visible through
        # the repository manifest.
        sources: list[CacheSource] = []
        products: tuple[str, ...] | None = None
        pending_by_source: dict[str, Path] = {}
        for file_path in pending_files:
            source, stage_products = inspect_stage_shard(str(file_path), self.stage)
            if products is None:
                products = stage_products
            elif products != stage_products:
                raise ValueError("Cache shards expose inconsistent stage schemas.")

            sources.append(source)
            pending_by_source[source.id] = file_path

        # Source ordering is stable across independently produced generations.
        sources.sort(key=lambda source: (source.file_name, source.id))
        source_tuple = tuple(sources)
        if (
            not self.transaction.parallel
            and self.transaction.snapshot.sources
            and self.transaction.snapshot.sources != source_tuple
        ):
            raise ValueError(
                "Cache stage does not use the published cache's ordered source set."
            )

        # A common source identity and event count are insufficient when a
        # filter compacted entries differently. Compare the persisted mapping
        # against one already published generation before making this visible.
        if self.transaction.snapshot.stages:
            reference_name, reference = next(
                iter(self.transaction.snapshot.stages.items())
            )
            for source in source_tuple:
                if source.id not in reference.shards:
                    continue
                pending_axis = read_source_entry_index(
                    str(pending_by_source[source.id]), self.stage
                )
                reference_path = self.repository.resolve_shard(
                    reference.shards[source.id]
                )
                reference_axis = read_source_entry_index(reference_path, reference_name)
                if not np.array_equal(pending_axis, reference_axis):
                    raise ValueError(
                        "Cache stage source-entry axis does not match the "
                        f"published cache for source '{source.file_name}'."
                    )

        self.transaction.publish(
            pending_by_source,
            source_tuple,
            products or (),
        )

    def close(self) -> None:
        """Close physical handles and discard an unpublished transaction.

        Successfully published shards are immutable and remain untouched.
        Otherwise, cleanup removes only this writer's private pending
        generation directory.
        """
        self._writer.close()
        if not self.transaction.published:
            self.transaction.cleanup()
