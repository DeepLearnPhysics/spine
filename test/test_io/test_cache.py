"""Tests for manifest-backed cache repositories."""

import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from spine.io.cache import (
    CacheManifest,
    CacheRepository,
    CacheSource,
    CacheStage,
    collect_garbage,
)
from spine.io.cache.backend.hdf5.reader import (
    inspect_stage_shard,
    read_source_entry_index,
)
from spine.io.cache.maintenance import remove_retired_stages
from spine.io.dataset import CacheDataset, HDF5Dataset, MixedDataset
from spine.io.manager import IOManager
from spine.io.read import CacheReader
from spine.io.write import CacheWriter, HDF5Writer
from spine.utils.conditional import TORCH_AVAILABLE


def cache_batch(values, source="raw.root", entries=None, key="value"):
    """Build a small writer batch with complete source provenance."""
    values = np.asarray(values)
    entries = np.arange(len(values)) if entries is None else np.asarray(entries)
    return {
        "index": np.arange(len(values)),
        "source_file_name": np.asarray([source] * len(values)),
        "source_file_size": np.asarray([123] * len(values)),
        "source_file_mtime_ns": np.asarray([456] * len(values)),
        "source_file_entry_index": entries,
        key: values,
    }


def write_stage(path, stage, key, values, overwrite=False):
    """Write and publish one test stage."""
    writer = CacheWriter(
        path=str(path),
        stage=stage,
        keys=[key],
        overwrite_stage=overwrite,
    )
    writer(cache_batch(values, key=key), {"base": {"iterations": 2}})
    writer.finalize()
    writer.finalize()
    writer.close()


def test_cache_round_trip_and_manifest_snapshot(tmp_path):
    """Independent stage shards should compose without rewriting predecessors."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "first", "x", [10, 11])
    first_manifest = CacheRepository(str(path)).load()
    first_shard = first_manifest.stages["first"].shards.copy()

    write_stage(path, "second", "y", [20, 21])
    manifest = CacheRepository(str(path)).load()
    assert manifest.generation == 2
    assert manifest.stages["first"].shards == first_shard
    assert set(manifest.stages) == {"first", "second"}
    repository = CacheRepository(str(path))
    first_shard_path = repository.resolve_shard(
        manifest.stages["first"].shards[manifest.sources[0].id]
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o2775
    assert stat.S_IMODE(repository.shard_dir.stat().st_mode) == 0o2775
    assert stat.S_IMODE(repository.manifest_path.stat().st_mode) == 0o664
    assert stat.S_IMODE(repository.lock_path.stat().st_mode) == 0o664
    assert stat.S_IMODE(Path(first_shard_path).stat().st_mode) == 0o664

    reader = CacheReader(path=str(path), keys=["x", "y"])
    assert reader[1]["x"] == 11
    assert reader[1]["y"] == 21
    assert [entry["source_file_entry_index"] for entry in reader.get_many([0, 1])] == [
        0,
        1,
    ]
    reader.close()

    reader = CacheReader(path=str(path))
    assert set(reader[0]) >= {"x", "y", "source_file_entry_index"}
    reader.close()


def test_cache_overwrite_reclaims_replaced_generation(tmp_path):
    """Replacing a stage should publish new data and reclaim old shards."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1, 2])
    repository = CacheRepository(str(path))
    old_manifest = repository.load()
    old_shard = Path(
        repository.resolve_shard(
            next(iter(old_manifest.stages["stage"].shards.values()))
        )
    )

    write_stage(path, "stage", "x", [3, 4], overwrite=True)
    new_reader = CacheReader(path=str(path), stage="stage")
    assert new_reader[0]["x"] == 3
    new_reader.close()
    assert not old_shard.exists()


def test_parallel_cache_publication_builds_source_roster(tmp_path):
    """Disjoint array-task contributions should form one readable stage."""
    path = tmp_path / "parallel.spine-cache"
    writers = [
        CacheWriter(
            path=str(path),
            stage="first",
            keys=["x"],
            parallel=True,
            expected_sources=2,
        )
        for _ in range(2)
    ]
    writers[0](cache_batch([1], source="a.root", key="x"), {})
    writers[1](cache_batch([2], source="b.root", key="x"), {})
    writers[0].finalize()
    with pytest.raises(RuntimeError, match="parallel contributions"):
        CacheReader(path=str(path), stage="first")
    writers[1].finalize()
    for writer in writers:
        writer.close()

    manifest = CacheRepository(str(path)).load()
    assert len(manifest.sources) == 2
    assert manifest.stages["first"].complete
    reader = CacheReader(path=str(path), stage="first")
    assert sorted(reader[index]["x"] for index in range(len(reader))) == [1, 2]
    reader.close()


def test_parallel_downstream_stage_is_hidden_until_complete(tmp_path):
    """A partial downstream generation must not look readable."""
    path = tmp_path / "parallel.spine-cache"
    for source, value in (("a.root", 1), ("b.root", 2)):
        writer = CacheWriter(
            path=str(path),
            stage="first",
            keys=["x"],
            parallel=True,
            expected_sources=2,
        )
        writer(cache_batch([value], source=source, key="x"), {})
        writer.finalize()
        writer.close()

    dependencies = {
        "first": CacheRepository(str(path)).load().stages["first"].generation
    }
    writers = [
        CacheWriter(
            path=str(path),
            stage="second",
            keys=["y"],
            parallel=True,
            dependencies=dependencies,
            expected_sources=2,
        )
        for _ in range(2)
    ]
    writers[0](cache_batch([3], source="a.root", key="y"), {})
    writers[1](cache_batch([4], source="b.root", key="y"), {})
    writers[0].finalize()
    with pytest.raises(RuntimeError, match="parallel contributions"):
        CacheReader(path=str(path), stage="second")
    writers[1].finalize()
    for writer in writers:
        writer.close()

    reader = CacheReader(path=str(path), stage="second")
    assert sorted(reader[index]["y"] for index in range(len(reader))) == [3, 4]
    reader.close()


def test_cache_stage_overwrite_invalidates_descendants(tmp_path):
    """Replacing an upstream should remove its stale descendant payloads."""
    path = tmp_path / "lineage.spine-cache"
    write_stage(path, "first", "x", [1])
    first = CacheRepository(str(path)).load().stages["first"]
    writer = CacheWriter(
        path=str(path),
        stage="second",
        keys=["y"],
        dependencies={"first": first.generation},
    )
    writer(cache_batch([2], key="y"), {})
    writer.finalize()
    writer.close()

    repository = CacheRepository(str(path))
    old_manifest = repository.load()
    old_shards = [
        Path(repository.resolve_shard(relative))
        for stage in old_manifest.stages.values()
        for relative in stage.shards.values()
    ]
    write_stage(path, "first", "x", [3], overwrite=True)
    manifest = repository.load()
    assert set(manifest.stages) == {"first"}
    assert all(not shard.exists() for shard in old_shards)


def test_cache_failed_replacement_preserves_published_generation(tmp_path, monkeypatch):
    """A failed manifest commit must retain the existing stage and its data."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1])
    repository = CacheRepository(str(path))
    old_manifest = repository.load()
    old_shard = Path(
        repository.resolve_shard(
            next(iter(old_manifest.stages["stage"].shards.values()))
        )
    )

    writer = CacheWriter(
        path=str(path), stage="stage", keys=["x"], overwrite_stage=True
    )
    writer(cache_batch([2], key="x"), {})

    def fail_publication(_manifest):
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(writer.repository, "_write_manifest", fail_publication)
    with pytest.raises(OSError, match="simulated manifest failure"):
        writer.finalize()
    writer.close()

    assert repository.load() == old_manifest
    assert old_shard.is_file()


def test_cache_garbage_collection_is_conservative(tmp_path):
    """GC should remove only old unreachable generations and transactions."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1])
    repository = CacheRepository(str(path))

    orphan = repository.shard_dir / "old" / "dead"
    orphan.mkdir(parents=True)
    (orphan / "orphan.h5").write_bytes(b"orphan")
    abandoned = repository.pending_dir / "abandoned"
    abandoned.mkdir()
    (abandoned / ".activity").touch()
    (abandoned / "partial.h5").write_bytes(b"partial")

    # A fresh heartbeat protects both a pending transaction and files which
    # that transaction has already moved into its generation directory.
    active = repository.pending_dir / "active"
    active.mkdir()
    (active / ".activity").touch()
    active_shards = repository.shard_dir / "new" / "active"
    active_shards.mkdir(parents=True)
    (active_shards / "moving.h5").write_bytes(b"moving")

    old_time = time.time() - 120
    for stale in (orphan, abandoned):
        for item in (stale, *stale.rglob("*")):
            os.utime(item, (old_time, old_time))

    report = collect_garbage(repository, dry_run=True, min_age_seconds=60)
    assert report.dry_run
    assert report.shard_generations == ("shards/old/dead",)
    assert report.pending_transactions == ("pending/abandoned",)
    assert report.total_bytes == len(b"orphanpartial")
    assert orphan.exists() and abandoned.exists()

    report = collect_garbage(repository, min_age_seconds=60)
    assert not report.dry_run
    assert not orphan.exists() and not abandoned.exists()
    assert active.exists() and active_shards.exists()

    manifest = repository.load()
    live_shard = repository.resolve_shard(
        next(iter(manifest.stages["stage"].shards.values()))
    )
    assert Path(live_shard).is_file()
    with pytest.raises(ValueError, match="cannot be negative"):
        collect_garbage(repository, min_age_seconds=-1)


def test_cache_cleanup_reports_failures_and_rejects_unsafe_paths(tmp_path, monkeypatch):
    """Post-publication cleanup should warn and retain strict path scoping."""
    path = tmp_path / "train.spine-cache"
    repository = CacheRepository(str(path), create=True)
    shard = repository.shard_dir / "stage" / "old" / "source.h5"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"old")
    retired = CacheStage(
        "old",
        (),
        {"source": str(shard.relative_to(repository.path))},
    )
    original_unlink = Path.unlink

    def fail_unlink(candidate, *args, **kwargs):
        if candidate == shard:
            raise OSError("simulated cleanup failure")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.warns(RuntimeWarning, match="cleanup was incomplete"):
        remove_retired_stages(repository, (retired,), ())
    assert shard.is_file()

    unsafe = CacheStage("old", (), {"source": "../outside.h5"})
    with pytest.raises(ValueError, match="escapes the shard directory"):
        remove_retired_stages(repository, (unsafe,), ())


def test_cache_gc_accepts_missing_maintenance_roots(tmp_path):
    """An older repository without a pending root should remain collectable."""
    path = tmp_path / "train.spine-cache"
    repository = CacheRepository(str(path), create=True)
    repository.pending_dir.rmdir()

    report = collect_garbage(repository, dry_run=True, min_age_seconds=0)
    assert report.pending_transactions == ()


def test_cache_writer_rejects_duplicate_stage_before_writing(tmp_path):
    """An existing stage should fail before creating a write transaction."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1])
    pending_before = set((path / "pending").iterdir())

    with pytest.raises(RuntimeError, match="already published"):
        CacheWriter(path=str(path), stage="stage", keys=["x"])

    assert set((path / "pending").iterdir()) == pending_before


def test_cache_writer_rejects_misaligned_source_axis(tmp_path):
    """Stages with different filtered source entries cannot share a manifest."""
    path = tmp_path / "train.spine-cache"
    writer = CacheWriter(path=str(path), stage="first", keys=["x"])
    writer(cache_batch([1, 2], entries=[2, 5], key="x"), {})
    writer.finalize()
    writer.close()

    writer = CacheWriter(path=str(path), stage="second", keys=["y"])
    writer(cache_batch([3, 4], entries=[2, 6], key="y"), {})
    with pytest.raises(ValueError, match="source-entry axis"):
        writer.finalize()
    writer.close()


def test_cache_writer_rejects_changed_source_set(tmp_path):
    """Every stage generation must cover the same ordered source roster."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "first", "x", [1])
    writer = CacheWriter(path=str(path), stage="second", keys=["y"])
    writer(cache_batch([2, 3], key="y"), {})
    with pytest.raises(ValueError, match="ordered source set"):
        writer.finalize()
    writer.close()


def test_cache_writer_rejects_inconsistent_shard_schemas(tmp_path):
    """All source shards in a stage generation must expose one schema."""
    path = tmp_path / "train.spine-cache"
    writer = CacheWriter(path=str(path), stage="stage", keys=["x"])
    batch = cache_batch([1, 2], key="x")
    batch["source_file_name"] = np.asarray(["a.root", "b.root"])
    writer(batch, {})
    writer._writer.finalize()
    writer._writer.close()
    shard = sorted(writer.transaction.pending_path.glob("*.h5"))[0]
    with h5py.File(shard, "a") as output:
        output["stages/stage/products"].create_group("extra")
    with pytest.raises(ValueError, match="inconsistent stage schemas"):
        writer.finalize()
    writer.close()


def test_cache_shard_validation_and_legacy_source_axis(tmp_path):
    """Incomplete shards fail inspection and missing provenance uses positions."""
    path = tmp_path / "train.spine-cache"
    writer = CacheWriter(path=str(path), stage="stage", keys=["x"])
    writer(cache_batch([1, 2], key="x"), {})
    writer._writer.finalize()
    writer._writer.close()
    shard = sorted(writer.transaction.pending_path.glob("*.h5"))[0]
    with h5py.File(shard, "a") as output:
        del output["stages/stage/products/source_file_entry_index"]
        output["stages/stage/info"].attrs["complete"] = False
    assert read_source_entry_index(str(shard), "stage").tolist() == [0, 1]
    with pytest.raises(RuntimeError, match="is not complete"):
        inspect_stage_shard(str(shard), "stage")

    with h5py.File(shard, "a") as output:
        output["stages/stage/info"].attrs["complete"] = True
        del output["source"]
    with pytest.raises(ValueError, match="incomplete source provenance"):
        inspect_stage_shard(str(shard), "stage")
    writer.close()


def test_cache_writer_validates_required_configuration(tmp_path):
    """Repository path, stage name and actual shard output are mandatory."""
    with pytest.raises(ValueError, match="repository `path`"):
        CacheWriter(stage="stage")
    with pytest.raises(ValueError, match="`stage` name"):
        CacheWriter(path=str(tmp_path / "empty.spine-cache"))
    with pytest.raises(ValueError, match="positive `expected_sources`"):
        CacheWriter(
            path=str(tmp_path / "parallel.spine-cache"),
            stage="stage",
            parallel=True,
        )

    writer = CacheWriter(path=str(tmp_path / "empty.spine-cache"), stage="stage")
    with pytest.raises(RuntimeError, match="wrote no shards"):
        writer.finalize()
    writer.close()


def test_cache_reader_product_routing_errors(tmp_path):
    """Ambiguous, absent and invalid stage routes should fail explicitly."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "first", "x", [1])
    write_stage(path, "second", "x", [2])

    with pytest.raises(ValueError, match="multiple cache stages"):
        CacheReader(path=str(path), keys=["x"])
    with pytest.raises(KeyError, match="does not contain stage 'missing'"):
        CacheReader(path=str(path), stage="missing")
    with pytest.raises(KeyError, match="does not contain product 'y'"):
        CacheReader(path=str(path), stage="first", keys=["y"])
    with pytest.raises(KeyError, match="does not contain product 'y'"):
        CacheReader(path=str(path), stage_map={"y": "first"}, keys=["y"])
    with pytest.raises(KeyError, match="does not contain stage 'missing'"):
        CacheReader(path=str(path), stage_map={"x": "missing"}, keys=["x"])
    with pytest.raises(KeyError, match="No cache stage contains product 'y'"):
        CacheReader(path=str(path), keys=["y"])

    reader = CacheReader(
        path=str(path), stage_map={"x": "second"}, keys=["x"], entry_list=[0]
    )
    assert reader[0]["x"] == 2
    reader.process_entry_list(n_entry=1)
    assert len(reader) == 1
    reader.close()

    with pytest.raises(ValueError, match="exposed by multiple"):
        CacheReader._merge_entries({"a": {"x": 1}, "b": {"x": 2}})
    with pytest.raises(ValueError, match="disagree on event metadata"):
        CacheReader._merge_entries({"a": {"index": 1}, "b": {"index": 2}})


def test_cache_reader_rejects_misaligned_physical_stage_axes(tmp_path):
    """Reader initialization verifies the event axes behind the manifest."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "first", "x", [1, 2])
    write_stage(path, "second", "y", [3, 4])
    repository = CacheRepository(str(path))
    manifest = repository.load()
    second_shard = repository.resolve_shard(
        next(iter(manifest.stages["second"].shards.values()))
    )
    with h5py.File(second_shard, "a") as output:
        output["stages/second/events"].resize((1,))
    with pytest.raises(ValueError, match="misaligned event axes"):
        CacheReader(path=str(path), keys=["x", "y"])


def test_cache_repository_validates_manifest_and_shard_paths(tmp_path):
    """Malformed manifests and paths outside a repository are rejected."""
    path = tmp_path / "train.spine-cache"
    repository = CacheRepository(str(path), create=True)
    assert repository.load() == CacheManifest()

    with pytest.raises(ValueError, match="escapes the repository"):
        repository.resolve_shard("../outside.h5")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        repository.resolve_shard("shards/missing.h5")

    bad_payloads = [
        {"format": "other", "version": 1},
        {"format": "spine_cache", "version": 2},
    ]
    for payload in bad_payloads:
        repository.manifest_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="unsupported"):
            repository.load()

    repository.manifest_path.write_text("[]")
    with pytest.raises(TypeError, match="root must be"):
        repository.load()


def test_cache_manifest_and_repository_publication_validation(tmp_path):
    """Manifest models reject duplicate, incomplete and stale source records."""
    source = {
        "id": "source",
        "file_name": "raw.root",
        "file_size": 1,
        "file_mtime_ns": 2,
        "num_entries": 3,
    }
    with pytest.raises(ValueError, match="duplicate source IDs"):
        CacheManifest.from_dict(
            {
                "format": "spine_cache",
                "version": 1,
                "generation": 0,
                "sources": [source, source],
                "stages": {},
            }
        )
    with pytest.raises(ValueError, match="exactly one shard"):
        CacheManifest.from_dict(
            {
                "format": "spine_cache",
                "version": 1,
                "generation": 1,
                "sources": [source],
                "stages": {"stage": {"generation": "g", "products": [], "shards": {}}},
            }
        )

    path = tmp_path / "repo.spine-cache"
    repository = CacheRepository(str(path), create=True)
    source_record = CacheSource(**source)
    stage_record = CacheStage("g", (), {"source": "shards/fake.h5"})
    with pytest.raises(RuntimeError, match="manifest changed"):
        repository.publish_stage("stage", stage_record, (source_record,), 2)

    repository.publish_stage("stage", stage_record, (source_record,), 0)
    with pytest.raises(RuntimeError, match="already published"):
        repository.publish_stage("stage", stage_record, (source_record,), 1)

    changed = CacheSource(**{**source, "num_entries": 4})
    with pytest.raises(ValueError, match="same ordered source set"):
        repository.publish_stage(
            "other", CacheStage("h", (), {"source": "other.h5"}), (changed,), 1
        )


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (
            {"generation": "g", "products": [], "shards": {"other": "x"}},
            "unknown sources",
        ),
        (
            {
                "generation": "g",
                "products": [],
                "shards": {},
                "complete": False,
                "expected_sources": 0,
            },
            "invalid expected source count",
        ),
        (
            {"generation": "g", "products": [], "shards": {}, "complete": False},
            "no completion target",
        ),
        (
            {
                "generation": "g",
                "products": [],
                "shards": {"source": "x"},
                "expected_sources": 2,
            },
            "marked complete",
        ),
        (
            {
                "generation": "g",
                "products": [],
                "shards": {"source": "x"},
                "dependencies": {"missing": "old"},
            },
            "stale or missing dependency",
        ),
    ],
)
def test_cache_manifest_rejects_invalid_parallel_and_lineage_records(stage, message):
    """Malformed completion and lineage metadata must fail on snapshot load."""
    source = {
        "id": "source",
        "file_name": "raw.root",
        "file_size": 1,
        "file_mtime_ns": 2,
        "num_entries": 3,
    }
    payload = {
        "format": "spine_cache",
        "version": 1,
        "generation": 1,
        "sources": [source],
        "stages": {"stage": stage},
    }
    with pytest.raises(ValueError, match=message):
        CacheManifest.from_dict(payload)


def test_cache_repository_rejects_conflicting_parallel_publications(tmp_path):
    """Parallel contributions must agree on source, schema and lineage."""
    path = tmp_path / "conflicts.spine-cache"
    repository = CacheRepository(str(path), create=True)
    source_a = CacheSource("a", "a.root", 1, 2, 1)
    source_b = CacheSource("b", "b.root", 1, 2, 1)
    partial = CacheStage("g", ("x",), {"a": "a.h5"}, expected_sources=2)

    with pytest.raises(ValueError, match="cannot overwrite"):
        repository.publish_stage(
            "stage", partial, (source_a,), 0, overwrite=True, parallel=True
        )
    repository.publish_stage("stage", partial, (source_a,), 0, parallel=True)

    changed_a = CacheSource("a", "a.root", 9, 2, 1)
    with pytest.raises(ValueError, match="changed between publications"):
        repository.publish_stage("stage", partial, (changed_a,), 0, parallel=True)
    with pytest.raises(ValueError, match="contains 1"):
        repository.publish_stage(
            "other",
            CacheStage("h", (), {"a": "a2.h5"}, expected_sources=3),
            (source_a,),
            0,
            parallel=True,
        )
    with pytest.raises(ValueError, match="different schemas"):
        repository.publish_stage(
            "stage",
            CacheStage("h", ("y",), {"b": "b.h5"}, expected_sources=2),
            (source_b,),
            0,
            parallel=True,
        )
    with pytest.raises(ValueError, match="different lineage"):
        repository.publish_stage(
            "stage",
            CacheStage(
                "h",
                ("x",),
                {"b": "b.h5"},
                dependencies={"upstream": "g"},
                expected_sources=2,
            ),
            (source_b,),
            0,
            parallel=True,
        )
    with pytest.raises(ValueError, match="different source counts"):
        repository.publish_stage(
            "stage",
            CacheStage("h", ("x",), {"b": "b.h5"}, expected_sources=3),
            (source_b,),
            0,
            parallel=True,
        )


def test_cache_repository_rejects_invalid_parallel_roster_changes(tmp_path):
    """Completed, duplicated and oversized source contributions fail clearly."""
    source_a = CacheSource("a", "a.root", 1, 2, 1)
    source_b = CacheSource("b", "b.root", 1, 2, 1)

    complete_path = tmp_path / "complete.spine-cache"
    repository = CacheRepository(str(complete_path), create=True)
    complete = CacheStage("g", ("x",), {"a": "a.h5"}, expected_sources=1)
    repository.publish_stage("first", complete, (source_a,), 0, parallel=True)
    with pytest.raises(RuntimeError, match="already complete"):
        repository.publish_stage("first", complete, (source_a,), 0, parallel=True)

    oversized_path = tmp_path / "oversized.spine-cache"
    repository = CacheRepository(str(oversized_path), create=True)
    with pytest.raises(ValueError, match="more than its 1 expected"):
        repository.publish_stage(
            "first",
            CacheStage("g", ("x",), {"a": "a.h5", "b": "b.h5"}, expected_sources=1),
            (source_a, source_b),
            0,
            parallel=True,
        )

    roster_path = tmp_path / "roster.spine-cache"
    repository = CacheRepository(str(roster_path), create=True)
    repository.publish_stage(
        "first", CacheStage("g", (), {"a": "a.h5"}), (source_a,), 0
    )
    repository.publish_stage(
        "second", CacheStage("h", (), {"a": "a2.h5"}), (source_a,), 1
    )
    with pytest.raises(ValueError, match="first stage may extend"):
        repository.publish_stage(
            "first",
            CacheStage("i", (), {"b": "b.h5"}, expected_sources=2),
            (source_b,),
            0,
            parallel=True,
        )

    duplicate_path = tmp_path / "duplicate.spine-cache"
    repository = CacheRepository(str(duplicate_path), create=True)
    repository.publish_stage(
        "first",
        CacheStage("g", (), {"a": "a.h5", "b": "b.h5"}),
        (source_a, source_b),
        0,
    )
    partial = CacheStage("h", (), {"a": "a2.h5"}, expected_sources=2)
    repository.publish_stage("second", partial, (source_a,), 1, parallel=True)
    with pytest.raises(ValueError, match="repeats source shards"):
        repository.publish_stage("second", partial, (source_a,), 0, parallel=True)


def test_cache_repository_rejects_stale_dependencies(tmp_path):
    """A stage cannot publish after an upstream generation changes."""
    path = tmp_path / "stale.spine-cache"
    repository = CacheRepository(str(path), create=True)
    source = CacheSource("a", "a.root", 1, 2, 1)
    with pytest.raises(RuntimeError, match="dependency 'upstream' changed"):
        repository.publish_stage(
            "stage",
            CacheStage(
                "g",
                (),
                {"a": "a.h5"},
                dependencies={"upstream": "old"},
            ),
            (source,),
            0,
        )


def test_cache_repository_and_transaction_validate_paths(tmp_path):
    """Missing repositories and path-like stage names fail before writing."""
    with pytest.raises(FileNotFoundError, match="has no manifest"):
        CacheRepository(str(tmp_path / "missing.spine-cache"))
    nonempty = tmp_path / "nonempty.spine-cache"
    nonempty.mkdir()
    (nonempty / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="nonempty directory"):
        CacheRepository(str(nonempty), create=True)
    with pytest.raises(ValueError, match="path components"):
        CacheWriter(path=str(tmp_path / "cache.spine-cache"), stage="../bad")


def test_cache_reader_rejects_empty_repository(tmp_path):
    """An initialized repository has no readable event domain before publish."""
    path = tmp_path / "train.spine-cache"
    CacheRepository(str(path), create=True)
    with pytest.raises(ValueError, match="contains no stages"):
        CacheReader(path=str(path))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for datasets.")
def test_flat_hdf5_dataset_rejects_cache_stage_routing():
    """Stage selection belongs exclusively to the cache dataset contract."""
    with pytest.raises(ValueError, match="cache dataset"):
        HDF5Dataset(file_keys="unused.h5", stage="deghosting")
    with pytest.raises(ValueError, match="cache dataset"):
        HDF5Dataset(file_keys="unused.h5", stage_map={"data": "deghosting"})


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for datasets.")
def test_cache_dataset_builds_and_validates_schema_stage_routing(monkeypatch):
    """Parser products should route to one unambiguous cache stage."""
    import spine.io.dataset.cache as cache_dataset_module
    import spine.io.dataset.hdf5 as hdf5_dataset_module

    class DummyParser:
        tree_keys = ("raw",)

    class DummyReader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __len__(self):
            return 1

    monkeypatch.setattr(
        hdf5_dataset_module, "instantiate", lambda *args, **kwargs: DummyParser()
    )
    monkeypatch.setattr(cache_dataset_module, "CacheReader", DummyReader)

    dataset = CacheDataset(
        path="unused.spine-cache",
        dtype="float32",
        stage_map={"raw": "first"},
        schema={"data": {"parser": "tensor", "stage": "first"}},
    )
    assert dataset.reader.kwargs["stage_map"] == {"raw": "first"}
    assert dataset.reader.kwargs["keys"] == ("raw",)

    with pytest.raises(ValueError, match="Conflicting cache schema"):
        CacheDataset(
            path="unused.spine-cache",
            dtype="float32",
            stage_map={"raw": "first"},
            schema={"data": {"parser": "tensor", "stage": "second"}},
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for datasets.")
def test_cache_dataset_and_canonical_mixed_dataset(tmp_path):
    """The cache dataset should work directly and as a mixed child source."""
    primary_path = tmp_path / "primary.h5"
    primary_writer = HDF5Writer(
        file_name=str(primary_path), keys=["x"], format_version=2
    )
    primary_writer({"index": np.arange(2), "x": np.asarray([1, 2])}, {})
    primary_writer.finalize()
    primary_writer.close()
    source_stat = primary_path.stat()

    repository_path = tmp_path / "train.spine-cache"
    batch = cache_batch([3, 4], source=primary_path.name, key="y")
    batch["source_file_size"][:] = source_stat.st_size
    batch["source_file_mtime_ns"][:] = source_stat.st_mtime_ns
    writer = CacheWriter(path=str(repository_path), stage="cached", keys=["y"])
    writer(batch, {})
    writer.finalize()
    writer.close()

    cache = CacheDataset(path=str(repository_path), stage="cached")
    assert cache[0]["y"] == 3
    mixed = MixedDataset(
        primary={"name": "hdf5", "file_keys": str(primary_path)},
        cache={"path": str(repository_path), "stage": "cached"},
        dtype="float32",
    )
    assert mixed[1]["x"] == 2
    assert mixed[1]["y"] == 4


def test_io_manager_extends_input_cache_repository(tmp_path):
    """A cache writer without a path should inherit its input repository."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "first", "x", [1])
    manager = IOManager(
        reader={"name": "cache", "path": str(path), "stage": "first"},
        writer={"name": "cache", "stage": "second", "keys": ["y"]},
    )
    assert manager.writer.repository.path == path.resolve()
    assert manager.writer.transaction.dependencies == {
        "first": CacheRepository(str(path)).load().stages["first"].generation
    }
    manager.close(finalize=False)

    nested_reader = SimpleNamespace(name="cache")
    nested = object.__new__(IOManager)
    nested.reader = SimpleNamespace(name="larcv")
    nested.loader = SimpleNamespace(
        dataset=SimpleNamespace(cache=SimpleNamespace(reader=nested_reader))
    )
    assert nested._get_cache_reader() is nested_reader
    nested.reader = None
    nested.loader = None
    assert nested._get_cache_reader() is None
    assert nested.dataset_provenance() is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"primary": {"name": "hdf5"}, "cache": {"name": "cache"}},
            "explicit `dtype`",
        ),
        (
            {"primary": {}, "cache": {"name": "cache"}, "dtype": "float32"},
            "primary.*dataset `name`",
        ),
        (
            {
                "primary": {"name": "hdf5"},
                "cache": {"name": "hdf5"},
                "dtype": "float32",
            },
            "cache.name.*must be `cache`",
        ),
    ],
)
def test_mixed_dataset_validates_child_configuration(kwargs, message):
    """Mixed children must use explicit primary/cache dataset semantics."""
    with pytest.raises(ValueError, match=message):
        MixedDataset(**kwargs)
