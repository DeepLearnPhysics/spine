"""Tests for manifest-backed cache repositories."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from spine.io.cache import CacheManifest, CacheRepository, CacheSource, CacheStage
from spine.io.cache.backend.hdf5.reader import (
    inspect_stage_shard,
    read_source_entry_index,
)
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


def test_cache_overwrite_keeps_active_reader_snapshot(tmp_path):
    """Replacing a stage should not alter readers of its old generation."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1, 2])
    old_reader = CacheReader(path=str(path), stage="stage")

    write_stage(path, "stage", "x", [3, 4], overwrite=True)
    new_reader = CacheReader(path=str(path), stage="stage")
    assert old_reader[0]["x"] == 1
    assert new_reader[0]["x"] == 3
    old_reader.close()
    new_reader.close()


def test_cache_writer_rejects_duplicate_stage_and_cleans_pending(tmp_path):
    """An unpublished transaction should be removable without cache mutation."""
    path = tmp_path / "train.spine-cache"
    write_stage(path, "stage", "x", [1])

    writer = CacheWriter(path=str(path), stage="stage", keys=["x"])
    pending = writer.transaction.pending_path
    writer(cache_batch([2], key="x"), {})
    with pytest.raises(RuntimeError, match="already published"):
        writer.finalize()
    writer.close()
    assert not pending.exists()


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
    changed = CacheSource(**{**source, "num_entries": 4})
    with pytest.raises(ValueError, match="same ordered source set"):
        repository.publish_stage(
            "other", CacheStage("h", (), {"source": "other.h5"}), (changed,), 1
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
        cache={"name": "cache", "path": str(repository_path), "stage": "cached"},
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
        ({}, "explicit `dtype`"),
        ({"primary": {}, "larcv": {}, "cache": {}}, "either `primary`"),
        ({"primary": {}, "cache": {}, "hdf5": {}}, "either `cache`"),
        ({"cache": {}}, "requires a `primary`"),
        ({"primary": {}}, "requires a `cache`"),
        (
            {"primary": {}, "cache": {}, "cache_align_keys": {}, "hdf5_align_keys": {}},
            "either `cache_align_keys`",
        ),
        (
            {"primary": {}, "cache": {}, "cache_key_map": {}, "hdf5_key_map": {}},
            "either `cache_key_map`",
        ),
    ],
)
def test_mixed_dataset_validates_canonical_child_configuration(kwargs, message):
    """Canonical and legacy mixed child options cannot be ambiguous."""
    if kwargs:
        kwargs.setdefault("dtype", "float32")
    with pytest.raises(ValueError, match=message):
        MixedDataset(**kwargs)
