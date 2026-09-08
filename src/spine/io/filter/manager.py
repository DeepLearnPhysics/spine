"""Scan-cache lifecycle and consolidated entry-filter manifests."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml

from spine.config import load_config_file
from spine.config.loader import resolve_config_path
from spine.version import __version__

from .base import EntryInspector, SourceFingerprint, resolve_sources
from .larcv import LArCVEntryInspector

SCAN_FORMAT = "spine-entry-filter-scan"
SCAN_SCHEMA_VERSION = 1
MANIFEST_FORMAT = "spine-entry-filter"
MANIFEST_SCHEMA_VERSION = 1

INSPECTORS: dict[str, type[EntryInspector]] = {"larcv": LArCVEntryInspector}

__all__ = [
    "build_manifest",
    "eligible_cache_entries_from_manifest",
    "eligible_entries_from_manifest",
    "load_entry_filter",
    "load_filter_config",
    "scan_sources",
]


def _atomic_yaml_dump(data: Mapping[str, Any], output: str | Path) -> None:
    """Write a YAML artifact transactionally beside its final path."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(data), stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_text_dump(lines: Sequence[str], output: str | Path) -> None:
    """Write a newline-delimited text artifact transactionally."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("".join(f"{line}\n" for line in lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _normalized_measurements(
    measurements: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate and normalize the deliberately narrow measurement schema."""
    if not isinstance(measurements, Mapping) or len(measurements) == 0:
        raise ValueError("Filter configuration requires non-empty `measurements`.")

    normalized = {}
    for name, request in measurements.items():
        if not isinstance(name, str) or not isinstance(request, Mapping):
            raise TypeError("Each measurement must map a name to a configuration.")
        kind = request.get("kind")
        if kind != "product_size":
            raise ValueError(
                f"Measurement `{name}` has unsupported kind `{kind}`; expected "
                "`product_size`."
            )
        normalized[name] = {"kind": "product_size"}
    return normalized


def _normalized_filters(
    filters: Mapping[str, Any], measurements: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """Validate exclusive upper-bound predicates against measurements."""
    if not isinstance(filters, Mapping) or len(filters) == 0:
        raise ValueError("Filter configuration requires non-empty `filters`.")

    normalized = {}
    for name, predicate in filters.items():
        if name not in measurements:
            raise ValueError(f"Filter `{name}` has no matching measurement.")
        if not isinstance(predicate, Mapping) or set(predicate) != {"max_count"}:
            raise ValueError(
                f"Filter `{name}` must define exactly one `max_count` predicate."
            )
        value = predicate["max_count"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Filter `{name}` `max_count` must be a positive integer.")
        normalized[name] = {"max_count": value}
    return normalized


def load_filter_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load, compose and validate a standalone entry-filter configuration.

    File-backed configurations use SPINE's standard resolver and loader, so
    includes, search paths, environment expansion and metadata checks match
    the main ``spine`` command.

    Parameters
    ----------
    config : str, Path or mapping
        Configuration path/name or an already resolved mapping.

    Returns
    -------
    dict
        Normalized backend, measurement and filter configuration.
    """
    if isinstance(config, Mapping):
        loaded = dict(config)
    else:
        cfg_path = resolve_config_path(str(config), current_dir=os.getcwd())
        loaded = load_config_file(cfg_path)

    input_cfg = loaded.get("input")
    if not isinstance(input_cfg, Mapping):
        raise TypeError("Filter configuration requires an `input` mapping.")
    backend = input_cfg.get("name")
    if backend not in (*INSPECTORS, "auto"):
        raise ValueError(
            f"Unknown entry-filter input backend `{backend}`; expected one of "
            f"{sorted(INSPECTORS)} or `auto`."
        )

    measurements = _normalized_measurements(loaded.get("measurements", {}))
    filters = _normalized_filters(loaded.get("filters", {}), measurements)
    return {
        "input": {"name": backend},
        "measurements": measurements,
        "filters": filters,
    }


def _resolve_backend(name: str, sources: Sequence[str]) -> str:
    """Resolve explicit or conservative extension-based backend selection."""
    if name != "auto":
        return name
    if all(source.lower().endswith(".root") for source in sources):
        return "larcv"
    raise ValueError("Could not infer an entry-filter backend from all sources.")


def _record_path(cache_dir: str | Path, source: str) -> Path:
    """Return the collision-resistant scan-record path for one source."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    basename = Path(source).name or "remote"
    return Path(cache_dir) / f"{basename}.{digest}.yaml"


def _record_valid(
    record: Any,
    fingerprint: SourceFingerprint,
    backend: str,
    inspector_version: int,
    measurements: Mapping[str, Any],
) -> bool:
    """Return whether a scan record is complete and safe to reuse."""
    if not isinstance(record, Mapping):
        return False
    expected = {
        "format": SCAN_FORMAT,
        "schema_version": SCAN_SCHEMA_VERSION,
        "complete": True,
        "backend": backend,
        "inspector_version": inspector_version,
        "source": fingerprint.to_dict(),
        "measurement_spec": dict(measurements),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return False

    num_entries = record.get("num_entries")
    values = record.get("measurements")
    return (
        isinstance(num_entries, int)
        and num_entries >= 0
        and isinstance(values, Mapping)
        and set(values) == set(measurements)
        and all(
            isinstance(column, list)
            and len(column) == num_entries
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in column
            )
            for column in values.values()
        )
    )


def _load_yaml(path: Path) -> Any:
    """Load a YAML artifact, returning ``None`` for a missing file."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        return None


def _scan_one(
    backend: str,
    source: str,
    measurements: Mapping[str, Mapping[str, Any]],
    cache_dir: str,
    force: bool,
) -> tuple[str, str, bool]:
    """Validate or regenerate one per-file counter record."""
    inspector = INSPECTORS[backend]()
    fingerprint = inspector.fingerprint(source)
    record_path = _record_path(cache_dir, fingerprint.path)
    existing = _load_yaml(record_path)
    if not force and _record_valid(
        existing, fingerprint, backend, inspector.version, measurements
    ):
        return fingerprint.path, str(record_path), True

    num_entries, values = inspector.inspect(fingerprint.path, measurements)
    record = {
        "format": SCAN_FORMAT,
        "schema_version": SCAN_SCHEMA_VERSION,
        "complete": True,
        "backend": backend,
        "inspector_version": inspector.version,
        "source": fingerprint.to_dict(),
        "num_entries": num_entries,
        "measurement_spec": dict(measurements),
        "measurements": values,
    }
    if not _record_valid(record, fingerprint, backend, inspector.version, measurements):
        raise RuntimeError(f"Inspector produced an invalid record for {source}.")
    _atomic_yaml_dump(record, record_path)
    return fingerprint.path, str(record_path), False


def scan_sources(
    config: str | Path | Mapping[str, Any],
    *,
    sources: str | Sequence[str] | None = None,
    source_list: str | None = None,
    cache_dir: str | Path,
    workers: int = 1,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Scan source files, reusing every compatible persistent record.

    Returns one summary per canonical source with its counter-record path and
    whether the record was reused.  Independent files are scanned in worker
    processes when ``workers`` is greater than one, keeping ROOT state isolated.

    Parameters
    ----------
    config : str, Path or mapping
        Filter configuration path/name or resolved mapping.
    sources : str or sequence[str], optional
        Direct source paths or glob expressions.
    source_list : str, optional
        Text file containing source expressions.
    cache_dir : str or Path
        Directory holding persistent per-source scan records.
    workers : int, default 1
        Number of source files to inspect concurrently.
    force : bool, default False
        Regenerate records even when their identity and schema are valid.

    Returns
    -------
    list[dict]
        Scan summaries in canonical source order.
    """
    if workers < 1:
        raise ValueError("`workers` must be at least one.")
    cfg = load_filter_config(config)

    # Any inspector can perform generic local/glob resolution.  Resolve first,
    # then make auto-selection authoritative for the complete collection.
    resolved = resolve_sources(sources, source_list)
    backend = _resolve_backend(cfg["input"]["name"], resolved)
    canonical = resolve_sources(resolved, None)
    cache_path = str(Path(cache_dir).resolve())
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    arguments = [
        (backend, source, cfg["measurements"], cache_path, force)
        for source in canonical
    ]
    if workers == 1:
        results = [_scan_one(*args) for args in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_scan_star, arguments))

    return [
        {"source": source, "record": record, "reused": reused}
        for source, record, reused in results
    ]


def _scan_star(arguments: tuple[Any, ...]) -> tuple[str, str, bool]:
    """Unpack one process-pool scan request."""
    return _scan_one(*arguments)


def _load_required_record(
    inspector: EntryInspector,
    source: str,
    cache_dir: str | Path,
    measurements: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], SourceFingerprint]:
    """Load and strictly validate one record required for finalization."""
    fingerprint = inspector.fingerprint(source)
    record_path = _record_path(cache_dir, fingerprint.path)
    record = _load_yaml(record_path)
    if not _record_valid(
        record,
        fingerprint,
        inspector.name,
        inspector.version,
        measurements,
    ):
        raise RuntimeError(
            f"Missing, stale or incomplete scan record for `{fingerprint.path}`: "
            f"{record_path}"
        )
    return record_path, record, fingerprint


def build_manifest(
    config: str | Path | Mapping[str, Any],
    *,
    sources: str | Sequence[str] | None = None,
    source_list: str | None = None,
    cache_dir: str | Path,
    output: str | Path,
    output_source_list: str | Path,
) -> dict[str, Any]:
    """Build and atomically write a consolidated entry-filter manifest.

    Every expected scan record must already exist and still match its source.
    Each ``max_count`` predicate is exclusive: values strictly below the bound
    are accepted, while values equal to or above it are rejected.

    Parameters
    ----------
    config : str, Path or mapping
        Filter configuration path/name or resolved mapping.
    sources : str or sequence[str], optional
        Direct source paths or glob expressions.
    source_list : str, optional
        Text file containing source expressions.
    cache_dir : str or Path
        Directory containing completed per-source scan records.
    output : str or Path
        Destination consolidated manifest.
    output_source_list : str or Path
        Destination list containing sources with accepted entries.

    Returns
    -------
    dict
        Consolidated manifest written to ``output``.
    """
    cfg = load_filter_config(config)
    resolved = resolve_sources(sources, source_list)
    backend = _resolve_backend(cfg["input"]["name"], resolved)
    inspector = INSPECTORS[backend]()
    canonical = resolve_sources(resolved, None)

    source_records = []
    total_entries = 0
    total_rejected = 0
    source_digest = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()
    for source in canonical:
        record_path, record, fingerprint = _load_required_record(
            inspector, source, cache_dir, cfg["measurements"]
        )
        num_entries = int(record["num_entries"])
        rejected = []
        for entry in range(num_entries):
            if any(
                record["measurements"][name][entry] >= predicate["max_count"]
                for name, predicate in cfg["filters"].items()
            ):
                rejected.append(entry)

        accepted = num_entries - len(rejected)
        total_entries += num_entries
        total_rejected += len(rejected)
        source_records.append(
            {
                **fingerprint.to_dict(),
                "num_entries": num_entries,
                "accepted_entries": accepted,
                "rejected_entries": rejected,
                "scan_record": str(record_path.resolve()),
            }
        )

    manifest = {
        "format": MANIFEST_FORMAT,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input": {"name": backend},
        "inspector_version": inspector.version,
        "spine_version": __version__,
        "source_collection_sha256": source_digest,
        "measurements": cfg["measurements"],
        "filters": cfg["filters"],
        "comparison": "count < max_count",
        "total_entries": total_entries,
        "accepted_entries": total_entries - total_rejected,
        "rejected_entries": total_rejected,
        "sources": source_records,
    }
    _atomic_text_dump(
        [record["path"] for record in source_records if record["accepted_entries"] > 0],
        output_source_list,
    )
    # Publish the manifest last so its presence certifies that all companion
    # build artifacts were written successfully.
    _atomic_yaml_dump(manifest, output)
    return manifest


def load_entry_filter(path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level structure of an entry-filter manifest.

    Parameters
    ----------
    path : str or Path
        Manifest path to load.

    Returns
    -------
    dict
        Validated manifest mapping.
    """
    manifest_path = Path(path).expanduser()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, Mapping):
        raise TypeError("Entry-filter manifest must be a mapping.")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("File is not a SPINE entry-filter manifest.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported entry-filter schema `{manifest.get('schema_version')}`."
        )
    if not isinstance(manifest.get("sources"), list):
        raise TypeError("Entry-filter manifest requires a `sources` list.")
    if not isinstance(manifest.get("input"), Mapping):
        raise TypeError("Entry-filter manifest requires an `input` mapping.")
    return dict(manifest)


def eligible_cache_entries_from_manifest(
    path: str | Path,
    provenance: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Project a source manifest onto provenance-bearing cache entries.

    This path does not inspect cached physics products. Instead, it matches
    each physical cache entry to one source record in an existing LArCV
    manifest using the persisted source file name, size and modification time,
    then applies that record's rejected source-entry indexes.

    Parameters
    ----------
    path : str or Path
        Existing LArCV entry-filter manifest.
    provenance : sequence[mapping]
        Per-cache-entry source provenance. Each mapping must contain all four
        ``source_file_*`` fields exposed by :class:`spine.io.read.ReaderBase`.

    Returns
    -------
    list[int]
        Eligible entries in the cache's physical global entry domain.

    Raises
    ------
    ValueError
        If the manifest is not LArCV-backed, provenance is malformed, or one
        cache source matches multiple manifest records.
    KeyError
        If required provenance is absent or a cache source is not represented
        in the manifest.
    """
    manifest = load_entry_filter(path)
    backend = manifest.get("input", {}).get("name")
    if backend != "larcv":
        raise ValueError(
            "Cache provenance filtering currently requires a LArCV manifest, "
            f"got `{backend}`."
        )
    if manifest.get("inspector_version") != LArCVEntryInspector.version:
        raise ValueError(
            f"Entry-filter inspector version `{manifest.get('inspector_version')}` "
            f"does not match current `{LArCVEntryInspector.version}`."
        )

    # Index records by their persisted lightweight identity. Cache files store
    # a basename rather than the original absolute source path.
    records: dict[tuple[str, int | None, int | None], list[Mapping[str, Any]]] = {}
    rejected_by_path: dict[str, set[int]] = {}
    for record in manifest["sources"]:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise TypeError("Every manifest source must be a mapping with a path.")
        source_path = record["path"]
        size = _normalize_source_fingerprint(record.get("size"), "size")
        mtime = _normalize_source_fingerprint(record.get("mtime_ns"), "mtime_ns")
        identity = (os.path.basename(source_path), size, mtime)
        records.setdefault(identity, []).append(record)
        rejected_by_path[source_path] = _validated_rejected_entries(record)

    eligible = []
    for cache_entry, source in enumerate(provenance):
        missing = set(
            (
                "source_file_name",
                "source_file_size",
                "source_file_mtime_ns",
                "source_file_entry_index",
            )
        ) - set(source)
        if missing:
            raise KeyError(
                "Cannot apply a source manifest to a cache without complete "
                f"provenance; missing keys: {sorted(missing)}."
            )

        file_name = source["source_file_name"]
        if isinstance(file_name, bytes):
            file_name = file_name.decode()
        if not isinstance(file_name, str):
            raise TypeError("`source_file_name` must be a string.")
        size = _normalize_source_fingerprint(source["source_file_size"], "size")
        mtime = _normalize_source_fingerprint(
            source["source_file_mtime_ns"], "mtime_ns"
        )
        identity = (os.path.basename(file_name), size, mtime)
        candidates = records.get(identity, [])
        if len(candidates) == 0:
            raise KeyError(
                "Cache source is missing from the entry-filter manifest: "
                f"{identity[0]} (size={identity[1]}, mtime_ns={identity[2]})."
            )
        if len(candidates) > 1:
            paths = sorted(str(candidate["path"]) for candidate in candidates)
            raise ValueError(
                "Cache source provenance matches multiple manifest records: "
                f"{paths}."
            )

        record = candidates[0]
        source_entry = source["source_file_entry_index"]
        if (
            isinstance(source_entry, bool)
            or not isinstance(source_entry, Integral)
            or source_entry < 0
            or source_entry >= record["num_entries"]
        ):
            raise ValueError(
                f"Invalid source entry `{source_entry}` for manifest source "
                f"`{record['path']}`."
            )
        if int(source_entry) not in rejected_by_path[str(record["path"])]:
            eligible.append(cache_entry)

    return eligible


def _normalize_source_fingerprint(value: Any, name: str) -> int | None:
    """Normalize local integers and remote-source sentinel fingerprints."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Source fingerprint `{name}` must be an integer or None.")
    value = int(value)
    return None if value < 0 else value


def _validated_rejected_entries(record: Mapping[str, Any]) -> set[int]:
    """Validate and return one manifest record's rejected entry indexes."""
    count = record.get("num_entries")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError(f"Invalid entry count for `{record.get('path')}`.")
    rejected = record.get("rejected_entries")
    if not isinstance(rejected, list) or any(
        isinstance(entry, bool)
        or not isinstance(entry, int)
        or entry < 0
        or entry >= count
        for entry in rejected
    ):
        raise ValueError(f"Invalid rejected entry list for `{record.get('path')}`.")
    if len(set(rejected)) != len(rejected):
        raise ValueError(f"Duplicate rejected entries for `{record.get('path')}`.")
    return set(rejected)


def eligible_entries_from_manifest(
    path: str | Path,
    *,
    backend: str,
    sources: Sequence[str],
    file_counts: Sequence[int],
) -> list[int]:
    """Resolve a manifest into eligible global entries for current sources.

    The manifest may describe a superset of the files currently being read,
    which lets one dataset-wide artifact serve scheduler jobs that each open a
    single source. Every current source must nevertheless have an exact path,
    fingerprint and entry-count match.

    Parameters
    ----------
    path : str or Path
        Consolidated entry-filter manifest.
    backend : str
        Reader backend consuming the manifest.
    sources : sequence[str]
        Current reader sources in canonical traversal order.
    file_counts : sequence[int]
        Number of entries contributed by each current source.

    Returns
    -------
    list[int]
        Eligible entries in the reader's global source domain.
    """
    if len(sources) != len(file_counts):
        raise ValueError("Source and file-count collections must have equal length.")
    manifest = load_entry_filter(path)
    if manifest.get("input", {}).get("name") != backend:
        raise ValueError(
            f"Entry-filter backend `{manifest.get('input', {}).get('name')}` does "
            f"not match reader backend `{backend}`."
        )

    inspector_cls = INSPECTORS.get(backend)
    if inspector_cls is None:
        raise ValueError(f"No entry-filter inspector is registered for `{backend}`.")
    inspector = inspector_cls()
    if manifest.get("inspector_version") != inspector.version:
        raise ValueError(
            f"Entry-filter inspector version `{manifest.get('inspector_version')}` "
            f"does not match current `{inspector.version}`."
        )
    records: dict[str, Mapping[str, Any]] = {}
    for record in manifest["sources"]:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise TypeError("Every manifest source must be a mapping with a path.")
        source_path = record["path"]
        if source_path in records:
            raise ValueError(f"Duplicate manifest source identity: {source_path}")
        records[source_path] = record

    eligible: list[int] = []
    offset = 0
    for source, count in zip(sources, file_counts):
        fingerprint = inspector.fingerprint(source)
        record = records.get(fingerprint.path)
        if record is None:
            raise KeyError(
                f"Current source is missing from entry-filter manifest: "
                f"{fingerprint.path}"
            )
        for key, value in fingerprint.to_dict().items():
            if record.get(key) != value:
                raise ValueError(
                    f"Entry-filter fingerprint mismatch for `{fingerprint.path}` "
                    f"field `{key}`."
                )
        if record.get("num_entries") != int(count):
            raise ValueError(
                f"Entry-filter count mismatch for `{fingerprint.path}`: manifest "
                f"has {record.get('num_entries')}, reader has {count}."
            )

        rejected = record.get("rejected_entries")
        if not isinstance(rejected, list) or any(
            isinstance(entry, bool)
            or not isinstance(entry, int)
            or entry < 0
            or entry >= count
            for entry in rejected
        ):
            raise ValueError(f"Invalid rejected entry list for `{fingerprint.path}`.")
        if len(set(rejected)) != len(rejected):
            raise ValueError(f"Duplicate rejected entries for `{fingerprint.path}`.")

        rejected_set = set(rejected)
        eligible.extend(
            offset + local_entry
            for local_entry in range(count)
            if local_entry not in rejected_set
        )
        offset += count

    return eligible
