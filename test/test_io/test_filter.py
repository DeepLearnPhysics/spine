"""Tests for reusable file-aware entry filtering."""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

import spine.io.filter.hdf5 as hdf5_filter_module
import spine.io.filter.larcv as larcv_filter_module
import spine.io.filter.manager as filter_manager
from spine.config import ConfigCycleError
from spine.data import EdgeIndexBatch
from spine.io.cache import CacheManifest, CacheStage
from spine.io.filter import (
    CacheEntryInspector,
    EntryInspector,
    HDF5EntryInspector,
    LArCVEntryInspector,
    build_manifest,
    eligible_entries_from_manifest,
    load_filter_config,
    scan_sources,
)
from spine.io.filter.base import canonical_source, resolve_sources
from spine.io.read import HDF5Reader
from spine.io.write import HDF5Writer
from spine.utils.conditional import LARCV_AVAILABLE, ROOT, ROOT_AVAILABLE, larcv


class FakeInspector(EntryInspector):
    """Read integer measurements from lightweight text sources."""

    name = "fake"
    version = 7
    calls: list[str] = []

    def inspect(self, source, measurements):
        self.calls.append(source)
        values = [int(value) for value in Path(source).read_text().split()]
        return len(values), {name: list(values) for name in measurements}


@pytest.fixture
def fake_backend(monkeypatch):
    """Register the deterministic test inspector for one test."""
    FakeInspector.calls = []
    monkeypatch.setitem(filter_manager.INSPECTORS, "fake", FakeInspector)
    return {
        "input": {"name": "fake"},
        "measurements": {"points": {"kind": "product_size"}},
        "filters": {"points": {"max_count": 500000}},
    }


def test_filter_config_uses_standard_composition(tmp_path, monkeypatch):
    """Filter configs should resolve nested composition and operations."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(config_dir))
    monkeypatch.setenv("FILTER_BACKEND", "larcv")
    (config_dir / "base.yaml").write_text(
        """
__meta__:
  kind: fragment
input:
  name: ${FILTER_BACKEND}
measurements:
  sparse3d_reco:
    kind: product_size
filters:
  sparse3d_reco:
    max_count: 500000
""",
        encoding="utf-8",
    )
    (config_dir / "middle.yaml").write_text(
        """
__meta__:
  kind: fragment
include: base.yaml
measurements:
  diagnostic:
    kind: product_size
filters:
  diagnostic:
    max_count: 100
""",
        encoding="utf-8",
    )
    (config_dir / "filter.yaml").write_text(
        """
include: middle.yaml
override:
  filters.sparse3d_reco.max_count: 500000
  measurements-: diagnostic
  filters-: diagnostic
""",
        encoding="utf-8",
    )

    config = load_filter_config("filter.yaml")

    assert config["filters"]["sparse3d_reco"]["max_count"] == 500000
    assert set(config["measurements"]) == {"sparse3d_reco"}


def test_filter_config_rejects_circular_includes(tmp_path):
    """The standalone filter loader should retain include-cycle checks."""
    (tmp_path / "one.yaml").write_text("include: two.yaml\n", encoding="utf-8")
    (tmp_path / "two.yaml").write_text("include: one.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigCycleError, match="Circular include"):
        load_filter_config(tmp_path / "one.yaml")


def test_filter_config_resolves_search_path_include(tmp_path, monkeypatch):
    """Includes should fall back to configured shared search directories."""
    local_dir = tmp_path / "local"
    shared_dir = tmp_path / "shared"
    local_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(shared_dir))
    (shared_dir / "shared.yaml").write_text(
        """
__meta__: {kind: fragment}
input: {name: larcv}
measurements:
  points: {kind: product_size}
filters:
  points: {max_count: 10}
""",
        encoding="utf-8",
    )
    (local_dir / "filter.yaml").write_text("include: shared.yaml\n", encoding="utf-8")

    config = load_filter_config(local_dir / "filter.yaml")

    assert config["filters"]["points"]["max_count"] == 10


def test_generic_source_resolution_and_fingerprints(tmp_path, fake_backend):
    """Generic source helpers should cover lists, globs and remote identities."""
    source = tmp_path / "input.dat"
    source.write_text("1", encoding="utf-8")
    source_list = tmp_path / "sources.txt"
    source_list.write_text(f"\n{source}\n{source}\n", encoding="utf-8")
    inspector = FakeInspector()

    assert resolve_sources(source_list=str(source_list)) == [str(source)]
    assert inspector.resolve_sources(str(source)) == [str(source)]
    assert resolve_sources("root://server/input.root") == ["root://server/input.root"]
    assert canonical_source("root://server/input.root") == "root://server/input.root"
    assert inspector.fingerprint("xroot://server/input.root").to_dict() == {
        "path": "xroot://server/input.root",
        "size": None,
        "mtime_ns": None,
    }
    with pytest.raises(ValueError, match="exactly one"):
        resolve_sources()
    with pytest.raises(ValueError, match="exactly one"):
        resolve_sources(str(source), str(source_list))
    with pytest.raises(FileNotFoundError, match="Source list"):
        resolve_sources(source_list=str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match="matched no files"):
        resolve_sources(str(tmp_path / "*.root"))

    empty_list = tmp_path / "empty.txt"
    empty_list.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No input sources"):
        resolve_sources(source_list=str(empty_list))


def test_entry_inspector_abstract_fallback():
    """The abstract inspection fallback should fail if explicitly delegated to."""

    class DelegatingInspector(EntryInspector):
        name = "delegate"

        def inspect(self, source, measurements):
            return super().inspect(source, measurements)

    with pytest.raises(NotImplementedError):
        DelegatingInspector().inspect("source", {})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"measurements": {}}, "measurements"),
        ({"input": {"name": "unknown"}}, "Unknown"),
        ({"filters": {"missing": {"max_count": 2}}}, "no matching"),
        ({"filters": {"points": {"max_count": 0}}}, "positive"),
    ],
)
def test_filter_config_validation(fake_backend, change, message):
    """Tool-specific validation should reject malformed resolved schemas."""
    config = {**fake_backend, **change}
    with pytest.raises((TypeError, ValueError), match=message):
        load_filter_config(config)


def test_filter_config_rejects_additional_schema_errors(fake_backend):
    """Measurement and predicate schemas should remain intentionally narrow."""
    cases = [
        ({**fake_backend, "input": []}, "input"),
        ({**fake_backend, "measurements": {1: {"kind": "product_size"}}}, "Each"),
        (
            {**fake_backend, "measurements": {"points": {"kind": "scalar"}}},
            "unsupported kind",
        ),
        ({**fake_backend, "filters": {}}, "non-empty `filters`"),
        ({**fake_backend, "filters": {"points": {"min_count": 1}}}, "exactly"),
        ({**fake_backend, "filters": {"points": {"max_count": True}}}, "positive"),
    ]
    for config, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            load_filter_config(config)


def test_filter_config_normalizes_hdf5_product_routing():
    """Cached-product requests should retain validated product and stage names."""
    config = load_filter_config(
        {
            "input": {"name": "cache"},
            "measurements": {
                "edges": {
                    "kind": "product_size",
                    "product": "fragment_graph_edge_index",
                    "stage": "fragment_graph",
                }
            },
            "filters": {"edges": {"max_count": 10}},
        }
    )

    assert config["measurements"]["edges"] == {
        "kind": "product_size",
        "product": "fragment_graph_edge_index",
        "stage": "fragment_graph",
    }
    for field, value in (("product", ""), ("stage", 1), ("unknown", "value")):
        request = {"kind": "product_size", field: value}
        with pytest.raises((TypeError, ValueError)):
            load_filter_config(
                {
                    "input": {"name": "cache"},
                    "measurements": {"edges": request},
                    "filters": {"edges": {"max_count": 10}},
                }
            )


def test_auto_backend_resolution(tmp_path):
    """Auto mode should infer only a uniformly supported source collection."""
    root_source = tmp_path / "input.root"
    data_source = tmp_path / "input.dat"
    root_source.touch()
    data_source.touch()
    hdf5_source = tmp_path / "input.h5"
    hdf5_source.touch()
    cache_source = tmp_path / "input.spine-cache"
    cache_source.mkdir()
    (cache_source / "manifest.json").write_text("{}", encoding="utf-8")

    assert filter_manager._resolve_backend("auto", [str(root_source)]) == "larcv"
    assert filter_manager._resolve_backend("auto", [str(hdf5_source)]) == "hdf5"
    assert filter_manager._resolve_backend("auto", [str(cache_source)]) == "cache"
    assert filter_manager._resolve_backend("larcv", [str(data_source)]) == "larcv"
    with pytest.raises(ValueError, match="Could not infer"):
        filter_manager._resolve_backend("auto", [str(data_source)])


@pytest.mark.parametrize("format_version", [1, 2])
def test_hdf5_product_filter_uses_event_row_counts(tmp_path, format_version):
    """HDF5 filtering should count product rows in both physical layouts."""
    source = tmp_path / f"cache-v{format_version}.h5"
    with HDF5Writer(
        str(source), overwrite=True, format_version=format_version
    ) as writer:
        writer(
            {
                "index": np.arange(3),
                "edges": [
                    np.zeros((2, 2)),
                    np.zeros((4, 2)),
                    np.zeros((1, 2)),
                ],
            },
            {},
        )

    config = {
        "input": {"name": "hdf5"},
        "measurements": {"shower_edges": {"kind": "product_size", "product": "edges"}},
        "filters": {"shower_edges": {"max_count": 4}},
    }
    cache_dir = tmp_path / "scan"
    scan_sources(config, sources=str(source), cache_dir=cache_dir)
    manifest = tmp_path / "accepted.yaml"
    build_manifest(
        config,
        sources=str(source),
        cache_dir=cache_dir,
        output=manifest,
        output_source_list=tmp_path / "sources.txt",
    )

    assert HDF5EntryInspector().inspect(str(source), config["measurements"]) == (
        3,
        {"shower_edges": [2, 4, 1]},
    )
    reader = HDF5Reader(str(source), entry_filter=str(manifest))
    np.testing.assert_array_equal(reader.entry_index, [0, 2])
    reader.close()


def test_hdf5_product_filter_counts_typed_edge_index_rows(tmp_path):
    """Typed edge indexes should be measured on their serialized edge axis."""
    source = tmp_path / "edge-index.h5"
    edge_index = EdgeIndexBatch(
        np.asarray(
            [
                [0, 0, 1, 1, 1, 1, 2],
                [0, 0, 1, 1, 1, 1, 2],
            ],
            dtype=np.int64,
        ),
        counts=[2, 4, 1],
        spans=[1, 1, 1],
        directed=True,
    )
    with HDF5Writer(str(source), overwrite=True, format_version=2) as writer:
        writer({"index": np.arange(3), "edges": edge_index}, {})

    # EdgeIndexBatch is canonicalized from in-memory (2, E) form to stored
    # (E, 2) rows. Event offsets must therefore advance by edge count, not 2.
    with h5py.File(source, "r") as in_file:
        offsets = in_file["products"]["edges"]["event_offsets"][:]
    np.testing.assert_array_equal(offsets, [0, 2, 6, 7])

    assert HDF5EntryInspector().inspect(
        str(source), {"edges": {"kind": "product_size"}}
    ) == (3, {"edges": [2, 4, 1]})


def test_hdf5_product_inspector_rejects_invalid_layouts(tmp_path):
    """Native inspection should fail clearly on missing or malformed metadata."""
    source = tmp_path / "cache.h5"
    with HDF5Writer(str(source), overwrite=True, format_version=2) as writer:
        writer(
            {"index": np.arange(2), "edges": [np.zeros((1, 2))] * 2},
            {},
        )
    inspector = HDF5EntryInspector()
    request = {"count": {"kind": "product_size", "product": "missing"}}
    with pytest.raises(KeyError, match="Missing requested"):
        inspector.inspect(str(source), request)

    with h5py.File(source, "r+") as out_file:
        offsets = out_file["products"]["edges"]["event_offsets"]
        offsets.resize((2,))
    request["count"]["product"] = "edges"
    with pytest.raises(ValueError, match="expected 2"):
        inspector.inspect(str(source), request)


@pytest.mark.parametrize(
    ("layout", "error", "message"),
    [
        ("missing_events", KeyError, "no event axis"),
        ("invalid_events", TypeError, "invalid event axis"),
        ("invalid_info", TypeError, "invalid info node"),
        ("missing_products", KeyError, "no product root"),
        ("invalid_products", TypeError, "invalid product root"),
        ("unsupported_version", ValueError, "Unsupported HDF5 format version"),
    ],
)
def test_hdf5_product_inspector_validates_file_roots(tmp_path, layout, error, message):
    """Native inspection should validate every required HDF5 root object."""
    source = tmp_path / f"{layout}.h5"
    with h5py.File(source, "w") as out_file:
        if layout != "missing_events":
            if layout == "invalid_events":
                out_file.create_group("events")
            else:
                out_file.create_dataset("events", data=np.arange(2))
        if layout == "invalid_info":
            out_file.create_dataset("info", data=np.arange(1))
        elif layout not in ("missing_events", "invalid_events"):
            info = out_file.create_group("info")
            info.attrs["format_version"] = 3 if layout == "unsupported_version" else 2
        if layout == "invalid_products":
            out_file.create_dataset("products", data=np.arange(1))
        elif layout not in (
            "missing_events",
            "invalid_events",
            "invalid_info",
            "missing_products",
            "unsupported_version",
        ):
            out_file.create_group("products")

    request = {"edges": {"kind": "product_size"}}
    with pytest.raises(error, match=message):
        HDF5EntryInspector().inspect(str(source), request)


@pytest.mark.parametrize(
    ("layout", "error", "message"),
    [
        ("product_dataset", TypeError, "does not expose"),
        ("offset_group", TypeError, "invalid `event_offsets`"),
        ("nonmonotonic", ValueError, "non-monotonic"),
    ],
)
def test_hdf5_product_inspector_validates_v2_offsets(tmp_path, layout, error, message):
    """V2 inspection should reject malformed product-offset metadata."""
    source = tmp_path / f"{layout}.h5"
    with h5py.File(source, "w") as out_file:
        out_file.create_dataset("events", data=np.arange(2))
        info = out_file.create_group("info")
        info.attrs["format_version"] = 2
        products = out_file.create_group("products")
        if layout == "product_dataset":
            products.create_dataset("edges", data=np.arange(1))
        else:
            edges = products.create_group("edges")
            if layout == "offset_group":
                edges.create_group("event_offsets")
            else:
                edges.create_dataset("event_offsets", data=[0, 2, 1])

    request = {"edges": {"kind": "product_size"}}
    with pytest.raises(error, match=message):
        HDF5EntryInspector().inspect(str(source), request)


def test_hdf5_product_inspector_validates_legacy_products(tmp_path):
    """Legacy inspection should require a referenced product dataset."""
    source = tmp_path / "legacy.h5"
    with HDF5Writer(str(source), overwrite=True, format_version=1) as writer:
        writer({"index": np.arange(1), "edges": [np.zeros((1, 2))]}, {})

    inspector = HDF5EntryInspector()
    with pytest.raises(KeyError, match="Missing requested"):
        inspector.inspect(str(source), {"missing": {"kind": "product_size"}})

    with h5py.File(source, "r+") as out_file:
        del out_file["edges"]
        out_file.create_group("edges")
    with pytest.raises(TypeError, match="not a dataset"):
        inspector.inspect(str(source), {"edges": {"kind": "product_size"}})


def test_hdf5_product_inspector_validates_direct_requests(tmp_path):
    """Inspector boundaries should defensively reject invalid product names."""
    source = tmp_path / "cache.h5"
    with HDF5Writer(str(source), overwrite=True, format_version=2) as writer:
        writer({"index": np.arange(1), "edges": [np.zeros((1, 2))]}, {})

    with pytest.raises(TypeError, match="nonempty string"):
        HDF5EntryInspector().inspect(
            str(source), {"edges": {"kind": "product_size", "product": ""}}
        )


def test_hdf5_filter_helpers_validate_physical_types(tmp_path):
    """Nested cache traversal should reject groups and datasets interchangeably."""
    source = tmp_path / "objects.h5"
    with h5py.File(source, "w") as out_file:
        out_file.create_dataset("dataset", data=np.arange(1))
        group = out_file.create_group("group")
        group.create_group("nested_group")
        with pytest.raises(TypeError, match="HDF5 group"):
            hdf5_filter_module._require_group(out_file, "dataset")
        with pytest.raises(TypeError, match="HDF5 dataset"):
            hdf5_filter_module._require_dataset(group, "nested_group")


def test_cache_product_inspector_validates_manifest_routing(tmp_path):
    """Cache product ownership should be explicit, valid, and unambiguous."""
    inspector = CacheEntryInspector()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        inspector.fingerprint(str(tmp_path / "missing.spine-cache"))

    manifest = CacheManifest(
        stages={
            "first": CacheStage("one", ("edges",), {}),
            "second": CacheStage("two", ("edges",), {}),
        }
    )
    cases = [
        ({"product": "edges", "stage": 1}, TypeError, "nonempty string"),
        ({"product": "edges"}, ValueError, "exactly one"),
        ({"product": "edges", "stage": "missing"}, KeyError, "no stage"),
        ({"product": "other", "stage": "first"}, KeyError, "no product"),
    ]
    for request, error, message in cases:
        with pytest.raises(error, match=message):
            inspector._resolve_owners(
                manifest, {"measurement": {"kind": "product_size", **request}}
            )


def test_hdf5_reader_rejects_cache_filter_manifest(tmp_path):
    """A flat HDF5 reader should reject manifests for another backend."""
    source = tmp_path / "cache.h5"
    with HDF5Writer(str(source), overwrite=True, format_version=2) as writer:
        writer({"index": np.arange(1), "value": np.arange(1)}, {})
    manifest = tmp_path / "cache-filter.yaml"
    manifest.write_text(
        "format: spine-entry-filter\nschema_version: 1\n"
        "input: {name: cache}\nsources: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot apply"):
        HDF5Reader(str(source), entry_filter=str(manifest))


def test_scan_reuses_and_invalidates_records(tmp_path, fake_backend):
    """Source fingerprints and measurement specs should control cache reuse."""
    source = tmp_path / "events.dat"
    source.write_text("1 2 3", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    first = scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)
    second = scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)
    assert not first[0]["reused"]
    assert second[0]["reused"]
    assert len(FakeInspector.calls) == 1

    source.write_text("1 2 3 4", encoding="utf-8")
    changed = scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)
    assert not changed[0]["reused"]
    assert len(FakeInspector.calls) == 2

    expanded = {
        **fake_backend,
        "measurements": {
            **fake_backend["measurements"],
            "diagnostic": {"kind": "product_size"},
        },
    }
    changed = scan_sources(expanded, sources=str(source), cache_dir=cache_dir)
    assert not changed[0]["reused"]
    assert len(FakeInspector.calls) == 3


def test_scan_force_and_incomplete_recovery(tmp_path, fake_backend):
    """Force should rescan valid records and unrelated temporary files be ignored."""
    source = tmp_path / "events.dat"
    source.write_text("1", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / ".interrupted.tmp").write_text("complete: false", encoding="utf-8")

    scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)
    result = scan_sources(
        fake_backend, sources=str(source), cache_dir=cache_dir, force=True
    )

    assert not result[0]["reused"]
    assert len(FakeInspector.calls) == 2


def test_scan_validation_and_parallel_dispatch(tmp_path, fake_backend, monkeypatch):
    """Worker validation and process-pool dispatch should be explicit."""
    source = tmp_path / "events.dat"
    source.write_text("1", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        scan_sources(fake_backend, sources=str(source), cache_dir=tmp_path, workers=0)

    calls = []

    class ImmediatePool:
        def __init__(self, max_workers):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def map(function, arguments):
            return map(function, arguments)

    monkeypatch.setattr(filter_manager, "ProcessPoolExecutor", ImmediatePool)
    result = scan_sources(
        fake_backend,
        sources=str(source),
        cache_dir=tmp_path / "cache",
        workers=2,
    )
    assert calls == [2]
    assert result[0]["source"] == str(source)


@pytest.mark.parametrize("values", [[1], [1, -1], [1, "two"], [1, True]])
def test_invalid_inspector_output_is_not_cached(
    tmp_path, fake_backend, monkeypatch, values
):
    """Malformed backend measurements should fail before publication."""

    class InvalidInspector(FakeInspector):
        def inspect(self, source, measurements):
            return 2, {"points": values}

    monkeypatch.setitem(filter_manager.INSPECTORS, "fake", InvalidInspector)
    source = tmp_path / "events.dat"
    source.write_text("1", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid record"):
        scan_sources(fake_backend, sources=str(source), cache_dir=tmp_path / "cache")


def test_malformed_scan_record_is_regenerated(tmp_path, fake_backend):
    """Unreadable YAML records should be treated as stale cache entries."""
    source = tmp_path / "events.dat"
    source.write_text("1", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    first = scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)
    Path(first[0]["record"]).write_text("[unterminated", encoding="utf-8")

    second = scan_sources(fake_backend, sources=str(source), cache_dir=cache_dir)

    assert not second[0]["reused"]


def test_scan_record_names_avoid_basename_collisions(tmp_path, fake_backend):
    """Equal basenames from different directories should receive distinct records."""
    sources = []
    for directory in (tmp_path / "one", tmp_path / "two"):
        directory.mkdir()
        source = directory / "events.dat"
        source.write_text("1", encoding="utf-8")
        sources.append(str(source))

    records = scan_sources(fake_backend, sources=sources, cache_dir=tmp_path / "cache")

    assert len({record["record"] for record in records}) == 2


def test_build_manifest_threshold_order_and_empty_sources(tmp_path, fake_backend):
    """The exclusive threshold and canonical source ordering should be exact."""
    first = tmp_path / "z.dat"
    second = tmp_path / "a.dat"
    first.write_text("499999 500000 500001", encoding="utf-8")
    second.write_text("500000", encoding="utf-8")
    sources = [str(first), str(second)]
    cache_dir = tmp_path / "cache"
    scan_sources(fake_backend, sources=sources, cache_dir=cache_dir)

    output = tmp_path / "filter.yaml"
    source_output = tmp_path / "filtered.txt"
    manifest = build_manifest(
        fake_backend,
        sources=list(reversed(sources)),
        cache_dir=cache_dir,
        output=output,
        output_source_list=source_output,
    )

    assert [record["path"] for record in manifest["sources"]] == sorted(sources)
    records = {Path(record["path"]).name: record for record in manifest["sources"]}
    assert records["z.dat"]["rejected_entries"] == [1, 2]
    assert records["a.dat"]["rejected_entries"] == [0]
    assert manifest["total_entries"] == 4
    assert manifest["accepted_entries"] == 1
    assert manifest["rejected_entries"] == 3
    assert source_output.read_text(encoding="utf-8").splitlines() == [str(first)]
    assert not list(tmp_path.glob(".*.tmp"))


def test_build_refuses_missing_and_stale_records(tmp_path, fake_backend):
    """Finalization should never silently omit or trust incompatible scans."""
    source = tmp_path / "events.dat"
    source.write_text("1", encoding="utf-8")
    kwargs = {
        "config": fake_backend,
        "sources": str(source),
        "cache_dir": tmp_path / "cache",
        "output": tmp_path / "filter.yaml",
        "output_source_list": tmp_path / "sources.txt",
    }
    with pytest.raises(RuntimeError, match="Missing, stale or incomplete"):
        build_manifest(**kwargs)

    scan_sources(fake_backend, sources=str(source), cache_dir=kwargs["cache_dir"])
    source.write_text("1 2", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing, stale or incomplete"):
        build_manifest(**kwargs)


def test_atomic_writers_remove_temporary_files_on_failure(tmp_path, monkeypatch):
    """A failed final rename should not leave an incomplete temporary artifact."""
    monkeypatch.setattr(
        filter_manager.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("rename failed")),
    )
    with pytest.raises(OSError, match="rename failed"):
        filter_manager._atomic_yaml_dump({"value": 1}, tmp_path / "value.yaml")
    with pytest.raises(OSError, match="rename failed"):
        filter_manager._atomic_text_dump(["value"], tmp_path / "value.txt")
    assert list(tmp_path.iterdir()) == []


def test_atomic_filter_artifacts_are_collaboration_readable(tmp_path):
    """Secure temporary creation must not leave published scans user-only."""
    yaml_path = tmp_path / "nested" / "value.yaml"
    text_path = tmp_path / "nested" / "value.txt"
    old_umask = os.umask(0o077)
    try:
        filter_manager._atomic_yaml_dump({"value": 1}, yaml_path)
        filter_manager._atomic_text_dump(["value"], text_path)
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(yaml_path.stat().st_mode) == 0o664
    assert stat.S_IMODE(text_path.stat().st_mode) == 0o664
    assert stat.S_IMODE(yaml_path.parent.stat().st_mode) == 0o2775


def test_manifest_serves_full_collection_and_subsets(tmp_path, fake_backend):
    """One manifest should translate exclusions for full and array-task inputs."""
    sources = [tmp_path / "a.dat", tmp_path / "b.dat"]
    sources[0].write_text("1 500000 2", encoding="utf-8")
    sources[1].write_text("500001 3", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    scan_sources(
        fake_backend, sources=[str(path) for path in sources], cache_dir=cache_dir
    )
    manifest_path = tmp_path / "filter.yaml"
    build_manifest(
        fake_backend,
        sources=[str(path) for path in sources],
        cache_dir=cache_dir,
        output=manifest_path,
        output_source_list=tmp_path / "sources.txt",
    )

    full = eligible_entries_from_manifest(
        manifest_path,
        backend="fake",
        sources=[str(path) for path in sources],
        file_counts=[3, 2],
    )
    subset = eligible_entries_from_manifest(
        manifest_path,
        backend="fake",
        sources=[str(sources[1])],
        file_counts=[2],
    )

    assert full == [0, 2, 4]
    assert subset == [1]
    with pytest.raises(ValueError, match="count mismatch"):
        eligible_entries_from_manifest(
            manifest_path,
            backend="fake",
            sources=[str(sources[0])],
            file_counts=[2],
        )

    missing = tmp_path / "missing.dat"
    missing.write_text("1", encoding="utf-8")
    with pytest.raises(KeyError, match="missing from entry-filter"):
        eligible_entries_from_manifest(
            manifest_path,
            backend="fake",
            sources=[str(missing)],
            file_counts=[1],
        )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["size"] += 1
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        eligible_entries_from_manifest(
            manifest_path,
            backend="fake",
            sources=[str(sources[0])],
            file_counts=[3],
        )


@pytest.mark.parametrize(
    ("manifest", "backend", "sources", "counts", "message"),
    [
        ([], "fake", [], [], "must be a mapping"),
        ({}, "fake", [], [], "not a SPINE"),
        (
            {"format": "spine-entry-filter", "schema_version": 2, "sources": []},
            "fake",
            [],
            [],
            "Unsupported",
        ),
        (
            {"format": "spine-entry-filter", "schema_version": 1},
            "fake",
            [],
            [],
            "sources",
        ),
        (
            {"format": "spine-entry-filter", "schema_version": 1, "sources": []},
            "fake",
            [],
            [],
            "input",
        ),
    ],
)
def test_manifest_top_level_validation(
    tmp_path, fake_backend, manifest, backend, sources, counts, message
):
    """Malformed manifest envelopes should fail before source matching."""
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        eligible_entries_from_manifest(
            path, backend=backend, sources=sources, file_counts=counts
        )


def test_manifest_source_validation_errors(tmp_path, fake_backend):
    """Reader integration should reject ambiguous source and exclusion data."""
    source = tmp_path / "source.dat"
    source.write_text("1", encoding="utf-8")
    fingerprint = FakeInspector().fingerprint(str(source)).to_dict()
    base = {
        "format": "spine-entry-filter",
        "schema_version": 1,
        "input": {"name": "fake"},
        "inspector_version": FakeInspector.version,
        "sources": [{**fingerprint, "num_entries": 1, "rejected_entries": []}],
    }

    def check(manifest, message, *, backend="fake", sources=None, counts=None):
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        with pytest.raises((TypeError, ValueError), match=message):
            eligible_entries_from_manifest(
                path,
                backend=backend,
                sources=sources if sources is not None else [str(source)],
                file_counts=counts if counts is not None else [1],
            )

    check(base, "equal length", sources=[str(source)], counts=[])
    check(base, "does not match", backend="larcv")
    unknown = {**base, "input": {"name": "unregistered"}}
    check(
        unknown,
        "No entry-filter inspector",
        backend="unregistered",
        sources=[],
        counts=[],
    )
    check({**base, "sources": [1]}, "mapping with a path")
    check({**base, "sources": base["sources"] * 2}, "Duplicate")
    check({**base, "inspector_version": 6}, "inspector version")

    for rejected in ("bad", [True], [1], [0, 0]):
        manifest = {
            **base,
            "sources": [{**base["sources"][0], "rejected_entries": rejected}],
        }
        message = "Duplicate" if rejected == [0, 0] else "Invalid rejected"
        check(manifest, message)


class FakeProduct:
    """Minimal LArCV event-product proxy."""

    def __init__(self, count):
        self.count = count

    def size(self):
        return self.count


class FakeTree:
    """Minimal ROOT tree exposing one changing product branch."""

    def __init__(self, name, counts):
        self.name = name
        self.counts = counts
        self.product = FakeProduct(0)

    def GetEntries(self):
        return len(self.counts)

    def GetEntry(self, entry):
        self.product.count = self.counts[entry]

    def SetBranchStatus(self, *_args):
        raise AssertionError("The inspector must not change ROOT branch status.")

    def __getattr__(self, name):
        if name == f"{self.name}_branch":
            return self.product
        raise AttributeError(name)


class FakeRootFile:
    """Minimal ROOT file keyed by canonical LArCV tree names."""

    def __init__(self, trees):
        self.trees = trees

    def IsZombie(self):
        return False

    def Close(self):
        return None

    def __getattr__(self, name):
        try:
            return self.trees[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_larcv_inspector_counts_products_and_checks_trees(monkeypatch):
    """The LArCV backend should count products and enforce tree consistency."""
    trees = {
        "sparse3d_reco_tree": FakeTree("sparse3d_reco", [2, 5]),
        "particle_pcluster_tree": FakeTree("particle_pcluster", [1, 3]),
    }
    root_file = FakeRootFile(trees)
    monkeypatch.setattr(larcv_filter_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "larcv", SimpleNamespace(__name__="larcv"))
    monkeypatch.setattr(
        larcv_filter_module,
        "ROOT",
        SimpleNamespace(TFile=lambda *_args: root_file),
    )
    requests = {
        "sparse3d_reco": {"kind": "product_size"},
        "particle_pcluster": {"kind": "product_size"},
    }

    num_entries, values = LArCVEntryInspector().inspect("input.root", requests)

    assert LArCVEntryInspector.version == 2
    assert num_entries == 2
    assert values == {"sparse3d_reco": [2, 5], "particle_pcluster": [1, 3]}
    root_file.trees.pop("particle_pcluster_tree")
    with pytest.raises(KeyError, match="Missing requested"):
        LArCVEntryInspector().inspect("input.root", requests)


def test_larcv_inspector_rejects_inconsistent_counts(monkeypatch):
    """Requested LArCV trees must describe the same event domain."""
    root_file = FakeRootFile(
        {
            "first_tree": FakeTree("first", [1]),
            "second_tree": FakeTree("second", [1, 2]),
        }
    )
    monkeypatch.setattr(larcv_filter_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "larcv", SimpleNamespace(__name__="larcv"))
    monkeypatch.setattr(
        larcv_filter_module,
        "ROOT",
        SimpleNamespace(TFile=lambda *_args: root_file),
    )
    requests = {
        "first": {"kind": "product_size"},
        "second": {"kind": "product_size"},
    }

    with pytest.raises(ValueError, match="has 2 entries; expected 1"):
        LArCVEntryInspector().inspect("input.root", requests)


def test_larcv_inspector_runtime_and_product_errors(monkeypatch):
    """Missing runtimes, bad files, requests and branches should fail clearly."""
    inspector = LArCVEntryInspector()
    request = {"points": {"kind": "product_size"}}
    monkeypatch.setattr(larcv_filter_module, "ROOT_AVAILABLE", False)
    with pytest.raises(ImportError, match="ROOT"):
        inspector.inspect("input.root", request)

    monkeypatch.setattr(larcv_filter_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "LARCV_AVAILABLE", False)
    with pytest.raises(ImportError, match="larcv"):
        inspector.inspect("input.root", request)

    monkeypatch.setattr(larcv_filter_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_filter_module, "larcv", SimpleNamespace(__name__="larcv"))
    monkeypatch.setattr(
        larcv_filter_module,
        "ROOT",
        SimpleNamespace(TFile=lambda *_args: FakeRootFile({})),
    )
    root_file = larcv_filter_module.ROOT.TFile("input.root")
    root_file.IsZombie = lambda: True
    monkeypatch.setattr(larcv_filter_module.ROOT, "TFile", lambda *_args: root_file)
    with pytest.raises(OSError, match="Could not open"):
        inspector.inspect("input.root", request)

    tree = FakeTree("points", [1])
    root_file.IsZombie = lambda: False
    root_file.trees["points_tree"] = tree
    with pytest.raises(ValueError, match="unsupported kind"):
        inspector.inspect("input.root", {"points": {"kind": "scalar"}})

    tree.product = SimpleNamespace(count=0)
    with pytest.raises(TypeError, match="does not expose"):
        inspector.inspect("input.root", request)


@pytest.mark.skipif(
    not (ROOT_AVAILABLE and LARCV_AVAILABLE),
    reason="ROOT and LArCV are required to inspect a real LArCV file.",
)
def test_larcv_inspector_matches_direct_branch_access(larcv_data):
    """Inspector counts must match direct event-product branch access."""
    _ = larcv.__name__
    root_file = ROOT.TFile(larcv_data, "r")

    try:
        # Use the first canonical product tree in the integration fixture.
        tree_names = [
            key.GetName()
            for key in root_file.GetListOfKeys()
            if key.GetName().endswith("_tree")
        ]
        assert len(tree_names) > 0
        tree_name = tree_names[0]
        product_name = tree_name[: -len("_tree")]
        branch_name = f"{product_name}_branch"
        tree = getattr(root_file, tree_name)

        expected = []
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            expected.append(int(getattr(tree, branch_name).size()))
    finally:
        root_file.Close()

    num_entries, values = LArCVEntryInspector().inspect(
        larcv_data,
        {product_name: {"kind": "product_size"}},
    )

    assert LArCVEntryInspector.version == 2
    assert num_entries == len(expected)
    assert any(count > 0 for count in expected)
    assert values[product_name] == expected
