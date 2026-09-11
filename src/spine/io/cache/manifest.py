"""Validated manifest models for a sharded SPINE cache repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["CacheManifest", "CacheSource", "CacheStage"]


@dataclass(frozen=True)
class CacheSource:
    """Describe one immutable upstream source represented by the cache.

    Attributes
    ----------
    id : str
        Stable identifier derived from the source-file identity.
    file_name : str
        Original source-file name recorded by the input reader.
    file_size : int
        Source-file size in bytes when the cache was produced.
    file_mtime_ns : int
        Source-file modification timestamp in nanoseconds.
    num_entries : int
        Number of cache entries published for this source.
    """

    id: str
    file_name: str
    file_size: int
    file_mtime_ns: int
    num_entries: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheSource":
        """Build a source record from decoded JSON data.

        Parameters
        ----------
        data : dict
            JSON-compatible source mapping from a cache manifest.

        Returns
        -------
        CacheSource
            Normalized immutable source record.
        """
        return cls(
            id=str(data["id"]),
            file_name=str(data["file_name"]),
            file_size=int(data["file_size"]),
            file_mtime_ns=int(data["file_mtime_ns"]),
            num_entries=int(data["num_entries"]),
        )


@dataclass(frozen=True)
class CacheStage:
    """Describe one published generation of a logical processing stage.

    Attributes
    ----------
    generation : str
        Unique identifier for the immutable stage generation.
    products : tuple[str, ...]
        Public product names exposed by the stage.
    shards : dict[str, str]
        Source-ID-to-relative-shard-path mapping.
    dependencies : dict[str, str]
        Upstream stage generations consumed to produce this stage.
    complete : bool
        Whether the stage covers the complete repository source roster.
    expected_sources : int, optional
        Required shard count for coordinated parallel construction.
    """

    generation: str
    products: tuple[str, ...]
    shards: dict[str, str]
    dependencies: dict[str, str] = field(default_factory=dict)
    complete: bool = True
    expected_sources: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheStage":
        """Build a stage record from decoded JSON data.

        Parameters
        ----------
        data : dict
            JSON-compatible stage mapping from a cache manifest.

        Returns
        -------
        CacheStage
            Normalized immutable stage record.
        """
        return cls(
            generation=str(data["generation"]),
            products=tuple(str(key) for key in data["products"]),
            shards={str(key): str(value) for key, value in data["shards"].items()},
            dependencies={
                str(key): str(value)
                for key, value in data.get("dependencies", {}).items()
            },
            complete=bool(data.get("complete", True)),
            expected_sources=(
                int(data["expected_sources"])
                if data.get("expected_sources") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this stage.

        Returns
        -------
        dict[str, Any]
            Stage metadata with tuples converted to JSON arrays.
        """
        result = asdict(self)
        result["products"] = list(self.products)
        return result


@dataclass(frozen=True)
class CacheManifest:
    """Represent one immutable snapshot of a cache repository.

    Attributes
    ----------
    format : str, default "spine_cache"
        Format discriminator used to reject unrelated JSON documents.
    version : int, default 1
        Manifest schema version.
    generation : int, default 0
        Monotonic repository generation used for optimistic concurrency.
    sources : tuple[CacheSource, ...]
        Ordered source set shared by every published stage.
    stages : dict[str, CacheStage]
        Published stage records keyed by logical stage name.
    """

    format: str = "spine_cache"
    version: int = 1
    generation: int = 0
    sources: tuple[CacheSource, ...] = ()
    stages: dict[str, CacheStage] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheManifest":
        """Validate and build a manifest from decoded JSON data.

        Parameters
        ----------
        data : dict
            Decoded manifest root.

        Returns
        -------
        CacheManifest
            Validated immutable repository snapshot.

        Raises
        ------
        ValueError
            If the format, schema version, source identities, or per-stage
            shard coverage violate the manifest contract.
        """
        if data.get("format") != "spine_cache":
            raise ValueError("Cache manifest has an unsupported format.")
        if int(data.get("version", 0)) != 1:
            raise ValueError("Cache manifest has an unsupported format version.")

        # Source order defines the stable logical file axis used by all stages.
        sources = tuple(CacheSource.from_dict(item) for item in data["sources"])
        source_ids = [source.id for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Cache manifest contains duplicate source IDs.")

        stages = {
            str(name): CacheStage.from_dict(stage)
            for name, stage in data["stages"].items()
        }
        expected_sources = set(source_ids)
        for name, stage in stages.items():
            shard_sources = set(stage.shards)
            if not shard_sources.issubset(expected_sources):
                raise ValueError(
                    f"Cache stage '{name}' contains shards for unknown sources."
                )
            if stage.complete and shard_sources != expected_sources:
                raise ValueError(
                    f"Complete cache stage '{name}' does not provide exactly "
                    "one shard for every manifest source."
                )
            if stage.expected_sources is not None and stage.expected_sources < 1:
                raise ValueError(
                    f"Cache stage '{name}' has an invalid expected source count."
                )
            if not stage.complete and stage.expected_sources is None:
                raise ValueError(
                    f"Incomplete cache stage '{name}' has no completion target."
                )
            if (
                stage.complete
                and stage.expected_sources is not None
                and len(stage.shards) != stage.expected_sources
            ):
                raise ValueError(
                    f"Cache stage '{name}' is marked complete before reaching "
                    "its expected source count."
                )

        # Published lineage must point to the currently visible generation.
        for name, stage in stages.items():
            for dependency, generation in stage.dependencies.items():
                upstream = stages.get(dependency)
                if upstream is None or upstream.generation != generation:
                    raise ValueError(
                        f"Cache stage '{name}' has a stale or missing dependency "
                        f"on '{dependency}' generation '{generation}'."
                    )

        return cls(
            generation=int(data["generation"]),
            sources=sources,
            stages=stages,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation of the manifest.

        Returns
        -------
        dict[str, Any]
            Complete manifest snapshot suitable for deterministic JSON
            serialization.
        """
        return {
            "format": self.format,
            "version": self.version,
            "generation": self.generation,
            "sources": [asdict(source) for source in self.sources],
            "stages": {
                name: stage.to_dict() for name, stage in sorted(self.stages.items())
            },
        }
