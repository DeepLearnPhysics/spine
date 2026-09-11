"""Filesystem lifecycle for manifest-backed SPINE cache repositories."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from spine.utils.file import make_shared_directory, set_shared_file_permissions

from .maintenance import remove_retired_stages
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
            # The root must exist before its lock file can be opened. Everything
            # else is checked and initialized under that lock so scheduler-array
            # tasks may safely arrive at a new repository simultaneously.
            make_shared_directory(self.path, parents=True, exist_ok=True)
            with self._manifest_lock():
                if not self.manifest_path.exists():
                    managed_names = {
                        self.lock_path.name,
                        self.shard_dir.name,
                        self.pending_dir.name,
                    }
                    unexpected = [
                        item.name
                        for item in self.path.iterdir()
                        if item.name not in managed_names
                    ]
                    if len(unexpected) > 0:
                        raise ValueError(
                            "Cannot initialize cache repository in nonempty "
                            f"directory '{self.path}'."
                        )

                    make_shared_directory(self.shard_dir, exist_ok=True)
                    make_shared_directory(self.pending_dir, exist_ok=True)
                    self._write_manifest(CacheManifest())
                else:
                    # Repair missing empty management directories left by an
                    # interrupted initializer or deliberate maintenance.
                    make_shared_directory(self.shard_dir, exist_ok=True)
                    make_shared_directory(self.pending_dir, exist_ok=True)
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
        parallel: bool = False,
    ) -> CacheManifest:
        """Atomically publish one stage generation or parallel contribution.

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
        parallel : bool, default False
            Merge a disjoint source contribution into the logical stage. The
            first stage may establish the source roster incrementally; later
            stages remain incomplete until every established source is present.

        Returns
        -------
        CacheManifest
            Newly published manifest snapshot.

        Raises
        ------
        RuntimeError
            If the transaction conflicts with an incompatible publication, or
            the stage already exists without replacement/parallel permission.
        ValueError
            If source identity, schema, lineage or parallel mode conflicts
            with the currently published repository state.
        """
        if parallel and not stage.publication_id:
            raise ValueError("Parallel cache publication requires a publication ID.")

        with self._manifest_lock():
            current = self.load()
            existing = current.stages.get(stage_name)
            if current.generation != base_generation and not parallel:
                raise RuntimeError(
                    "Cache manifest changed while this stage was being written; "
                    "restart the stage against the latest cache snapshot."
                )
            if existing is not None and not (overwrite or parallel):
                raise RuntimeError(
                    f"Cache stage '{stage_name}' is already published. Set "
                    "overwrite_stage=True to replace it."
                )
            if not parallel and current.sources and current.sources != sources:
                raise ValueError(
                    "Published cache stages must use the same ordered source set."
                )

            if parallel:
                if (
                    existing is not None
                    and existing.complete
                    and existing.publication_id == stage.publication_id
                ):
                    # A late retry from an already completed array attempt is
                    # idempotent. Its newly moved shard is no longer needed.
                    self._validate_parallel_retry(
                        current, stage_name, existing, stage, sources
                    )
                    live = (*current.stages.values(), *current.replacements.values())
                    remove_retired_stages(self, (stage,), live)
                    return current
                if overwrite and existing is not None and existing.complete:
                    return self._publish_parallel_replacement(
                        current, stage_name, stage, sources
                    )

                parallel_retired: list[CacheStage] = []
                merge_existing = existing
                merge_current = current
                if (
                    existing is not None
                    and not existing.complete
                    and existing.publication_id != stage.publication_id
                ):
                    if not overwrite:
                        raise RuntimeError(
                            f"Parallel cache stage '{stage_name}' belongs to "
                            f"publication '{existing.publication_id}', not "
                            f"'{stage.publication_id}'. Set overwrite_stage=True "
                            "to restart it."
                        )
                    parallel_retired.append(existing)
                    merge_existing = None
                    if set(current.stages) == {stage_name}:
                        # The incomplete first stage owns the provisional
                        # repository roster. A new attempt rebuilds both.
                        merge_current = replace(current, sources=())

                stage, merged_sources, superseded = self._merge_parallel_stage(
                    merge_current,
                    stage_name,
                    stage,
                    sources,
                    existing=merge_existing,
                    use_active=len(parallel_retired) == 0,
                )
                if superseded is not None:
                    parallel_retired.append(superseded)
            else:
                merged_sources = current.sources or sources
                parallel_retired = []

            # Reject a stage built against lineage which changed while it ran.
            for dependency, generation in stage.dependencies.items():
                upstream = current.stages.get(dependency)
                if upstream is None or upstream.generation != generation:
                    raise RuntimeError(
                        f"Cache dependency '{dependency}' changed while stage "
                        f"'{stage_name}' was being written."
                    )

            # Construct a new immutable snapshot. Replacing an upstream stage
            # removes every transitive descendant from the visible manifest.
            stages = dict(current.stages)
            replacements = dict(current.replacements)
            retired: list[CacheStage] = parallel_retired
            if overwrite and existing is not None:
                stages, retired_stages = self._without_descendants(stages, stage_name)
                replacements, retired_replacements = (
                    self._without_affected_replacements(
                        replacements, set(retired_stages)
                    )
                )
                retired.extend(retired_stages.values())
                retired.extend(retired_replacements.values())
            stages[stage_name] = stage
            updated = replace(
                current,
                generation=current.generation + 1,
                sources=merged_sources,
                stages=stages,
                replacements=replacements,
            )
            self._write_manifest(updated)

            # Publication is now authoritative. Reclaim the exact replaced
            # generations while the lock still excludes another publisher.
            if len(retired) > 0:
                live = (*updated.stages.values(), *updated.replacements.values())
                remove_retired_stages(self, retired, live)
            return updated

    def _publish_parallel_replacement(
        self,
        current: CacheManifest,
        stage_name: str,
        contribution: CacheStage,
        sources: tuple[CacheSource, ...],
    ) -> CacheManifest:
        """Publish one hidden contribution toward a parallel replacement.

        Parameters
        ----------
        current : CacheManifest
            Manifest snapshot held under the repository publication lock.
        stage_name : str
            Active logical stage being replaced.
        contribution : CacheStage
            New source-shard contribution prepared by one array task.
        sources : tuple[CacheSource, ...]
            Source records represented by the contribution.

        Returns
        -------
        CacheManifest
            Snapshot containing either the still-hidden replacement or its
            atomically activated complete generation.

        Notes
        -----
        The active stage remains untouched until the replacement covers the
        full repository roster. Repeated source contributions replace only
        that source in the hidden candidate, allowing failed array tasks to be
        retried without exposing a partial generation.
        """
        candidate = current.replacements.get(stage_name)
        retired: list[CacheStage] = []
        if (
            candidate is not None
            and candidate.publication_id != contribution.publication_id
        ):
            # A new orchestrator attempt starts a clean hidden candidate. The
            # old one remains live until this manifest update commits.
            retired.append(candidate)
            candidate = None

        replacement, _, superseded = self._merge_parallel_stage(
            current,
            stage_name,
            contribution,
            sources,
            existing=candidate,
            use_active=False,
        )
        # Replacement lineage must remain valid for the entire array run.
        for dependency, generation in replacement.dependencies.items():
            upstream = current.stages.get(dependency)
            if upstream is None or upstream.generation != generation:
                raise RuntimeError(
                    f"Cache dependency '{dependency}' changed while replacement "
                    f"stage '{stage_name}' was being written."
                )

        stages = dict(current.stages)
        replacements = dict(current.replacements)
        if superseded is not None:
            retired.append(superseded)

        if replacement.complete:
            # The final contribution performs the only reader-visible cutover.
            stages, retired_stages = self._without_descendants(stages, stage_name)
            replacements.pop(stage_name, None)
            replacements, retired_replacements = self._without_affected_replacements(
                replacements, set(retired_stages)
            )
            stages[stage_name] = replacement
            retired.extend(retired_stages.values())
            retired.extend(retired_replacements.values())
        else:
            replacements[stage_name] = replacement

        updated = replace(
            current,
            generation=current.generation + 1,
            stages=stages,
            replacements=replacements,
        )
        self._write_manifest(updated)

        if len(retired) > 0:
            live = (*updated.stages.values(), *updated.replacements.values())
            remove_retired_stages(self, retired, live)
        return updated

    @staticmethod
    def _validate_parallel_retry(
        current: CacheManifest,
        stage_name: str,
        existing: CacheStage,
        contribution: CacheStage,
        sources: tuple[CacheSource, ...],
    ) -> None:
        """Validate a late contribution from an already completed attempt.

        Parameters
        ----------
        current : CacheManifest
            Current repository snapshot.
        stage_name : str
            Logical stage targeted by the late contribution.
        existing, contribution : CacheStage
            Published stage and redundant source contribution, respectively.
        sources : tuple[CacheSource, ...]
            Source records represented by the redundant contribution.

        Raises
        ------
        ValueError
            If the late task does not match the completed attempt's schema,
            lineage, source roster or expected source count.
        """
        if existing.products != contribution.products:
            raise ValueError("Parallel cache contributions expose different schemas.")
        if existing.dependencies != contribution.dependencies:
            raise ValueError("Parallel cache contributions have different lineage.")
        if existing.expected_sources != contribution.expected_sources:
            raise ValueError(
                "Parallel cache contributions expect different source counts."
            )

        current_sources = {source.id: source for source in current.sources}
        for source in sources:
            if source.id not in current_sources:
                raise ValueError(
                    f"Parallel cache stage '{stage_name}' contains an unknown "
                    f"source '{source.id}'."
                )
            if current_sources[source.id] != source:
                raise ValueError(
                    f"Cache source '{source.id}' changed between publications."
                )

    @staticmethod
    def _merge_parallel_stage(
        current: CacheManifest,
        stage_name: str,
        contribution: CacheStage,
        sources: tuple[CacheSource, ...],
        existing: CacheStage | None = None,
        use_active: bool = True,
    ) -> tuple[CacheStage, tuple[CacheSource, ...], CacheStage | None]:
        """Merge one disjoint source contribution into a stage under lock.

        The repository's first stage may establish the global source roster
        incrementally. Once another stage exists, that roster is immutable and
        parallel contributions can only fill the established source slots.

        Parameters
        ----------
        current : CacheManifest
            Manifest snapshot held under the repository publication lock.
        stage_name : str
            Logical stage receiving the contribution.
        contribution : CacheStage
            Stage metadata and shard paths produced by one writer task.
        sources : tuple[CacheSource, ...]
            Source records represented by the contribution.
        existing : CacheStage, optional
            Hidden replacement candidate to extend. If omitted, extend the
            currently visible stage with the same name.
        use_active : bool, default True
            Fall back to the visible stage when ``existing`` is omitted. This
            is disabled when starting a fresh hidden or restarted candidate.

        Returns
        -------
        CacheStage
            Combined logical stage with its completion state recomputed.
        tuple[CacheSource, ...]
            Stable merged repository source roster.
        CacheStage or None
            Superseded duplicate-source shard records to clean after commit.

        Raises
        ------
        RuntimeError
            If the logical stage is already complete.
        ValueError
            If source identity, schema, lineage, expected count or shard
            ownership conflicts with an earlier contribution.
        """
        if existing is None and use_active:
            existing = current.stages.get(stage_name)
        superseded = None
        current_sources = {source.id: source for source in current.sources}
        contribution_sources = {source.id: source for source in sources}

        # A repeated source ID must carry exactly the same immutable identity.
        overlap = set(current_sources).intersection(contribution_sources)
        for source_id in overlap:
            if current_sources[source_id] != contribution_sources[source_id]:
                raise ValueError(
                    f"Cache source '{source_id}' changed between publications."
                )

        new_ids = set(contribution_sources).difference(current_sources)
        if (
            current.sources
            and not new_ids
            and contribution.expected_sources != len(current.sources)
        ):
            raise ValueError(
                f"Parallel cache stage '{stage_name}' expects "
                f"{contribution.expected_sources} sources, but the repository "
                f"contains {len(current.sources)}."
            )
        if (
            new_ids
            and current.stages
            and not (len(current.stages) == 1 and stage_name in current.stages)
        ):
            raise ValueError(
                "Only parallel construction of the repository's first stage "
                "may extend its source roster."
            )

        if existing is not None:
            if existing.complete and not new_ids:
                raise RuntimeError(f"Cache stage '{stage_name}' is already complete.")
            if existing.products != contribution.products:
                raise ValueError(
                    "Parallel cache contributions expose different schemas."
                )
            if existing.dependencies != contribution.dependencies:
                raise ValueError("Parallel cache contributions have different lineage.")
            if existing.publication_id != contribution.publication_id:
                raise ValueError(
                    "Parallel cache contributions have different publication IDs."
                )
            if existing.expected_sources != contribution.expected_sources:
                raise ValueError(
                    "Parallel cache contributions expect different source counts."
                )
            duplicate_shards = set(existing.shards).intersection(contribution.shards)
            if duplicate_shards:
                superseded = replace(
                    existing,
                    shards={
                        source_id: existing.shards[source_id]
                        for source_id in duplicate_shards
                    },
                )
            contribution = replace(
                existing,
                shards={**existing.shards, **contribution.shards},
            )

        merged_sources = tuple(
            sorted(
                {**current_sources, **contribution_sources}.values(),
                key=lambda source: (source.file_name, source.id),
            )
        )
        contribution = replace(
            contribution,
            complete=(
                len(contribution.shards) == contribution.expected_sources
                and set(contribution.shards) == {source.id for source in merged_sources}
            ),
        )
        if len(contribution.shards) > int(contribution.expected_sources or 0):
            raise ValueError(
                f"Parallel cache stage '{stage_name}' received more than its "
                f"{contribution.expected_sources} expected source shards."
            )

        return contribution, merged_sources, superseded

    @staticmethod
    def _without_affected_replacements(
        replacements: dict[str, CacheStage], stale_names: set[str]
    ) -> tuple[dict[str, CacheStage], dict[str, CacheStage]]:
        """Remove hidden candidates invalidated by an active-stage cutover.

        Parameters
        ----------
        replacements : dict[str, CacheStage]
            In-progress replacements keyed by their logical target stage.
        stale_names : set[str]
            Active stage names removed by the cutover.

        Returns
        -------
        tuple[dict[str, CacheStage], dict[str, CacheStage]]
            Still-valid candidates followed by candidates whose target or
            lineage was invalidated.
        """
        retired = {
            name: stage
            for name, stage in replacements.items()
            if name in stale_names
            or any(dependency in stale_names for dependency in stage.dependencies)
        }
        kept = {
            name: stage for name, stage in replacements.items() if name not in retired
        }
        return kept, retired

    @staticmethod
    def _without_descendants(
        stages: dict[str, CacheStage], stage_name: str
    ) -> tuple[dict[str, CacheStage], dict[str, CacheStage]]:
        """Remove a replaced stage and all transitive lineage descendants.

        Parameters
        ----------
        stages : dict[str, CacheStage]
            Current visible stages keyed by logical name.
        stage_name : str
            Upstream stage whose generation is being replaced.

        Returns
        -------
        tuple[dict[str, CacheStage], dict[str, CacheStage]]
            Stages independent of the replacement followed by the retired
            stage and all of its transitive descendants.
        """
        stale = {stage_name}
        changed = True
        while changed:
            changed = False
            for name, record in stages.items():
                if name not in stale and any(
                    dependency in stale for dependency in record.dependencies
                ):
                    stale.add(name)
                    changed = True

        return (
            {name: stage for name, stage in stages.items() if name not in stale},
            {name: stage for name, stage in stages.items() if name in stale},
        )

    @contextmanager
    def _manifest_lock(self) -> Iterator[None]:
        """Hold an advisory exclusive lock around one manifest update.

        Yields
        ------
        None
            Control while the repository lock is held by this process.
        """
        lock_exists = self.lock_path.exists()
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            if not lock_exists:
                set_shared_file_permissions(self.lock_path)
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
        set_shared_file_permissions(temporary)
        os.replace(temporary, self.manifest_path)
