"""Tests for the HDF5 reader."""

import multiprocessing

import h5py
import numpy as np
import pytest
import yaml
from yaml.parser import ParserError

import spine.data
from spine.data import ObjectList, RecoParticle, RunInfo
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.read import HDF5Reader, StageHDF5Reader
from spine.io.read.hdf5.common import (
    decode_string_attribute,
    require_dataset,
    require_group,
)
from spine.io.read.hdf5.product import _ProductHandles
from spine.io.write import HDF5Writer, StageHDF5Writer


def _read_hdf5_entry(path, queue):
    """Read one HDF5 entry in a child process and report a stable subset."""
    reader = HDF5Reader(path, build_classes=False)
    entry = reader.get(0)
    queue.put((entry["index"], entry["file_entry_index"]))
    reader.close()


def test_hdf5_reader(hdf5_data):
    """Tests the loading of an HDF5 file."""
    # Get the list of tree keys in the HDF5 file
    data_keys = None
    with h5py.File(hdf5_data, "r") as h5_file:
        data_keys = list(h5_file.keys())
        num_entries = len(h5_file["events"])

    # Intialize the reader
    reader = HDF5Reader(hdf5_data, create_run_map=True, build_classes=False)

    # Check that the number of events in the dataset is as expected
    assert reader.num_entries == num_entries

    # Load every entry, check that they contain what is expected
    for entry in reader:
        for key in data_keys:
            if key not in ["info", "events"]:
                assert key in entry

    # Check that the run map exists
    assert reader.run_map is not None
    assert reader.num_entries == len(reader.run_map)

    # Check all the available entry restriction modes
    reader.process_entry_list(n_entry=2)
    assert len(reader) == 2

    reader.process_entry_list(n_skip=2)
    assert len(reader) == reader.num_entries - 2

    reader.process_entry_list(n_entry=3, n_skip=2)
    assert len(reader) == 3

    reader.process_entry_list(entry_list=[1, 3, 4])
    assert len(reader) == 3

    reader.process_entry_list(skip_entry_list=[1, 3, 4])
    assert len(reader) == reader.num_entries - 3

    reader.process_entry_list(run_event_list=[tuple(reader.run_info[0])])
    reader.get_run_event(*reader.run_info[0])
    assert len(reader) == 1

    reader.process_entry_list(skip_run_event_list=[tuple(reader.run_info[0])])
    reader.get_run_event(*reader.run_info[1])
    assert len(reader) == reader.num_entries - 1

    # Try loading a file list
    reader = HDF5Reader([hdf5_data, hdf5_data], build_classes=False)
    assert reader.num_entries == 2 * num_entries
    for _ in reader:  # forces loading of all entries
        pass

    # Check that the internal indexing makes sense
    assert len(reader.file_index) == reader.num_entries
    assert len(reader.file_offsets) == 2
    assert reader.file_offsets[0] == 0
    assert reader.file_offsets[1] == num_entries

    # Check that the internals do not get compromised on sequential
    # restrictions when using a file list
    reader.process_entry_list(n_entry=2)
    assert len(reader) == 2

    reader.process_entry_list(n_skip=2)
    assert len(reader) == reader.num_entries - 2

    # Try to restrict the number of files to be loaded
    reader = HDF5Reader([hdf5_data, hdf5_data], limit_num_files=1)
    assert reader.num_entries == num_entries


def test_hdf5_reader_requires_events_dataset(tmp_path):
    """Reader initialization should reject files with no event tree."""
    path = tmp_path / "missing_events.h5"
    with h5py.File(path, "w") as out_file:
        out_file.create_group("info")

    with pytest.raises(AssertionError, match="event tree"):
        HDF5Reader(str(path))


def test_hdf5_reader_rejects_incomplete_files(tmp_path):
    """Reader should reject files explicitly marked as incomplete by default."""
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["version"] = "test"
        info.attrs["cfg"] = "{}"
        info.attrs["complete"] = False
        out_file.create_dataset("events", data=np.empty(0, dtype=[("dummy", np.int64)]))

    with pytest.raises(RuntimeError, match="marked incomplete"):
        HDF5Reader(str(path))


def test_hdf5_reader_can_ignore_incomplete_files(tmp_path):
    """Reader should allow explicitly incomplete files when requested."""
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["version"] = "test"
        info.attrs["cfg"] = "{}"
        info.attrs["complete"] = False
        out_file.create_dataset(
            "events", data=np.asarray([(0,)], dtype=[("dummy", np.int64)])
        )

    reader = HDF5Reader(str(path), ignore_incomplete=True)
    assert len(reader) == 1
    reader.close()


def test_hdf5_reader_enables_run_map_for_run_event_filters(hdf5_data):
    """Run-event restrictions should force run-map creation during initialization."""
    reader = HDF5Reader(hdf5_data, run_event_list=[])
    assert reader.run_map is not None


def test_hdf5_reader_reuses_open_handles(monkeypatch, hdf5_data):
    """Repeated entry reads should reuse a persistent file handle by default."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.read.hdf5.reader.h5py.File", counted_file)
    reader = HDF5Reader(hdf5_data, build_classes=False)
    init_calls = open_calls

    reader.get(0)
    reader.get(0)
    assert open_calls - init_calls == 1

    reader.close()


def test_hdf5_reader_reopens_handles_after_pid_change(monkeypatch, hdf5_data):
    """Reader handles should be process-local and reopen after a PID change."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    pids = iter([100, 200])
    monkeypatch.setattr("spine.io.read.hdf5.reader.h5py.File", counted_file)
    monkeypatch.setattr("spine.io.read.hdf5.reader._get_reader_pid", lambda: next(pids))

    reader = HDF5Reader(hdf5_data, build_classes=False)
    init_calls = open_calls

    reader.get(0)
    reader.get(0)
    assert open_calls - init_calls == 2

    reader.close()


def test_hdf5_reader_supports_independent_concurrent_readers(hdf5_data):
    """Separate read-only reader instances should access the same file independently."""
    reader_a = HDF5Reader(hdf5_data, build_classes=False)
    reader_b = HDF5Reader(hdf5_data, build_classes=False)

    entry_a = reader_a.get(0)
    entry_b = reader_b.get(0)

    assert entry_a["index"] == entry_b["index"] == 0
    assert entry_a["file_entry_index"] == entry_b["file_entry_index"] == 0

    reader_a.close()
    reader_b.close()


def test_hdf5_reader_supports_concurrent_reads_across_processes(hdf5_data):
    """Separate processes should be able to read the same finished HDF5 file."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_read_hdf5_entry, args=(hdf5_data, queue)) for _ in range(2)
    ]

    for process in processes:
        process.start()

    results = [queue.get(timeout=10) for _ in processes]

    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert results == [(0, 0), (0, 0)]


def test_hdf5_reader_keep_open_false_opens_per_get(monkeypatch, hdf5_data):
    """Disabling persistent handles should reopen the file on every access."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.read.hdf5.reader.h5py.File", counted_file)
    reader = HDF5Reader(hdf5_data, build_classes=False, keep_open=False)
    init_calls = open_calls

    reader.get(0)
    reader.get(0)

    assert open_calls - init_calls == 2
    assert reader._file_handles == {}


def test_hdf5_reader_close_swallows_handle_close_errors():
    """Reader cleanup should clear state even if a handle raises on close."""
    reader = HDF5Reader.__new__(HDF5Reader)
    reader._file_handles = {}
    reader._handle_pid = 123

    class BadHandle:
        def close(self):
            raise OSError("boom")

    reader._file_handles[0] = BadHandle()
    reader.close()

    assert reader._file_handles == {}
    assert reader._handle_pid is None


def test_hdf5_reader_v2_round_trip_and_projection(tmp_path):
    """The public reader should auto-detect V2 and project products at I/O time."""
    path = tmp_path / "v2.h5"
    particle = RecoParticle(
        id=4,
        index=np.asarray([1, 3, 8], dtype=np.int32),
        match_ids=np.asarray([12], dtype=np.int32),
        match_overlaps=np.asarray([0.5], dtype=np.float32),
    )
    data = {
        "index": np.asarray([0, 1]),
        "run_info": [RunInfo(run=1, event=10), RunInfo(run=1, event=11)],
        "meta": [
            ImageMeta3D(
                lower=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                upper=np.asarray([4.0, 6.0, 8.0], dtype=np.float32),
                size=np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
                count=np.asarray([4, 3, 4], dtype=np.int64),
            ),
            ImageMeta3D(),
        ],
        "particles": [
            ObjectList([particle], RecoParticle()),
            ObjectList([], RecoParticle()),
        ],
        "tensor": [
            np.arange(6, dtype=np.float32).reshape(3, 2),
            np.ones((1, 2), dtype=np.float32),
        ],
        "label": ["first", "second"],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(data, cfg={"format": 2})

    reader = HDF5Reader(str(path), create_run_map=True)
    assert reader.file_format_versions == [2]
    assert reader.run_map == {(1, -1, 10): 0, (1, -1, 11): 1}
    first = reader.get(0)
    second = reader.get(1)
    assert first["label"] == "first"
    assert first["particles"][0].size == 3
    assert np.array_equal(first["particles"][0].index, [1, 3, 8])
    assert np.array_equal(first["meta"].index_multipliers, [12, 4, 1])
    assert second["particles"] == []
    assert second["tensor"].shape == (1, 2)
    assert reader._product_object_schemas
    assert reader._product_object_handles
    assert reader._product_handles
    reader.close()
    assert reader._product_object_handles == {}
    assert reader._product_handles == {}

    # Closing a persistent reader invalidates cached h5py objects. A later
    # access must transparently rebuild those process-local handle caches.
    assert reader.get(0)["particles"][0].size == 3
    assert reader._product_object_handles
    reader.close()

    raw_reader = HDF5Reader(str(path), keys=["particles"], build_classes=False)
    assert isinstance(raw_reader.get(0)["particles"][0], dict)
    raw_reader.close()

    fixed_reader = HDF5Reader(
        str(path),
        keys=["particles"],
        build_classes=False,
        fixed_only=True,
    )
    fixed_particle = fixed_reader.get(0)["particles"][0]
    assert fixed_particle["id"] == 4
    assert fixed_particle["size"] == 3
    assert fixed_particle["best_match_id"] == 12
    assert fixed_particle["best_match_overlap"] == pytest.approx(0.5)
    assert "index" not in fixed_particle
    assert "match_ids" not in fixed_particle
    assert all(
        not pool_values
        for _, _, pool_values in fixed_reader._product_object_handles.values()
    )
    fixed_reader.close()

    fixed_class_reader = HDF5Reader(str(path), keys=["particles"], fixed_only=True)
    fixed_class_particle = fixed_class_reader.get(0)["particles"][0]
    assert fixed_class_particle.id == 4
    assert len(fixed_class_particle.index) == 0
    assert fixed_class_particle.size == 0
    fixed_class_reader.close()

    projected = HDF5Reader(str(path), keys=["tensor"])
    entry = projected.get(0)
    assert "tensor" in entry
    assert "particles" not in entry
    assert "label" not in entry
    projected.close()

    columnar = HDF5Reader(
        str(path),
        columnar=True,
        chunk_size=2,
    )
    columnar.configure_columnar(
        {
            "particles": (
                ("best_match_id", "best_match_overlap", "id", "size"),
                True,
            )
        }
    )
    chunk = columnar.get_columnar(0)
    assert chunk["index"].tolist() == [0, 1]
    assert chunk["particles"]["event_offsets"].tolist() == [0, 1, 1]
    assert chunk["particles"]["id"].tolist() == [4]
    assert chunk["particles"]["best_match_id"].tolist() == [12]
    assert chunk["particles"]["best_match_overlap"].tolist() == [0.5]
    columnar.close()


def test_hdf5_reader_v1_columnar_projection(tmp_path):
    """Legacy region references should support fixed-field projection."""
    path = tmp_path / "v1_columnar.h5"
    particle = RecoParticle(
        id=3,
        pid=2,
        index=np.asarray([1, 2], dtype=np.int32),
        match_ids=np.asarray([0], dtype=np.int32),
        match_overlaps=np.asarray([0.75], dtype=np.float32),
    )
    data = {
        "index": np.asarray([0]),
        "particles": [ObjectList([particle], RecoParticle())],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=1) as writer:
        writer(data, cfg={})

    reader = HDF5Reader(str(path), columnar=True)
    reader.configure_columnar(
        {
            "particles": (
                ("best_match_id", "best_match_overlap", "id", "pid", "size"),
                True,
            )
        }
    )
    chunk = reader.get_columnar(0)

    assert chunk["particles"]["event_offsets"].tolist() == [0, 1]
    assert chunk["particles"]["id"].tolist() == [3]
    assert chunk["particles"]["size"].tolist() == [2]
    assert chunk["particles"]["best_match_id"].tolist() == [0]
    reader.close()


@pytest.mark.parametrize("format_version", [1, 2])
def test_hdf5_reader_columnar_offsets_only(tmp_path, format_version):
    """Empty field projections should return boundaries without compound I/O."""
    path = tmp_path / f"offsets_only_v{format_version}.h5"
    data = {
        "index": np.asarray([0, 1]),
        "particles": [
            ObjectList([RecoParticle(id=3)], RecoParticle()),
            ObjectList([], RecoParticle()),
        ],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=format_version) as writer:
        writer(data, cfg={})

    reader = HDF5Reader(str(path), columnar=True)
    reader.configure_columnar({"particles": ((), True)})
    product = reader.get_columnar(0)["particles"]

    assert list(product) == ["event_offsets"]
    assert product["event_offsets"].tolist() == [0, 1, 1]
    reader.close()


def test_hdf5_reader_validates_columnar_state_and_bounds(tmp_path):
    """Columnar APIs should reject invalid mode, size, state and chunk IDs."""
    path = tmp_path / "columnar_validation.h5"
    data = {
        "index": np.asarray([0]),
        "particles": [ObjectList([], RecoParticle())],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(data, cfg={})

    with pytest.raises(ValueError, match="positive integer"):
        HDF5Reader(str(path), columnar=True, chunk_size=0)

    event_reader = HDF5Reader(str(path))
    with pytest.raises(RuntimeError, match="event mode"):
        event_reader.configure_columnar({"particles": (("id",), True)})
    with pytest.raises(RuntimeError, match="not enabled"):
        event_reader.get_columnar(0)
    event_reader.close()

    reader = HDF5Reader(str(path), columnar=True)
    with pytest.raises(IndexError, match="out of bounds"):
        reader.get_columnar(1)
    with pytest.raises(RuntimeError, match="not configured"):
        reader.get_columnar(0)
    reader.close()


def test_hdf5_reader_columnar_missing_products_and_fields(tmp_path):
    """Columnar projections should distinguish optional and required data."""
    path = tmp_path / "columnar_missing.h5"
    data = {
        "index": np.asarray([0]),
        "particles": [
            ObjectList([RecoParticle(id=3)], RecoParticle()),
        ],
        "tensor": [np.ones((1, 2), dtype=np.float32)],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(data, cfg={})

    optional = HDF5Reader(str(path), columnar=True, keep_open=False)
    optional.configure_columnar({"absent": (("id",), False)})
    assert "absent" not in optional.get_columnar(0)
    optional.close()

    required = HDF5Reader(str(path), columnar=True)
    required.configure_columnar({"absent": (("id",), True)})
    with pytest.raises(KeyError, match="Required columnar product"):
        required.get_columnar(0)
    required.close()

    missing_field = HDF5Reader(str(path), columnar=True)
    missing_field.configure_columnar({"particles": (("not_a_field",), True)})
    with pytest.raises(KeyError, match="missing fixed fields"):
        missing_field.get_columnar(0)
    missing_field.close()

    wrong_kind = HDF5Reader(str(path), columnar=True)
    wrong_kind.configure_columnar({"tensor": (None, True)})
    with pytest.raises(TypeError, match="object collection"):
        wrong_kind.get_columnar(0)
    wrong_kind.close()


def test_hdf5_reader_columnar_run_splitting_and_legacy_errors(tmp_path):
    """Run splitting and malformed legacy projections should be explicit."""
    assert HDF5Reader._contiguous_runs(np.empty(0, dtype=np.int64)) == []
    assert HDF5Reader._contiguous_runs(np.asarray([1, 2, 4, 7, 8])) == [
        (1, 3),
        (4, 5),
        (7, 9),
    ]

    path = tmp_path / "legacy_errors.h5"
    data = {
        "index": np.asarray([0]),
        "particles": [
            ObjectList([RecoParticle(id=3)], RecoParticle()),
        ],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=1) as writer:
        writer(data, cfg={})

    missing_field = HDF5Reader(str(path), columnar=True)
    missing_field.configure_columnar({"particles": (("not_a_field",), True)})
    with pytest.raises(KeyError, match="Region-reference product"):
        missing_field.get_columnar(0)
    missing_field.close()

    with h5py.File(path, "a") as out_file:
        out_file.create_dataset(
            "orphan",
            data=np.asarray([(4,)], dtype=[("id", np.int64)]),
        )

    missing_reference = HDF5Reader(str(path), columnar=True)
    missing_reference.configure_columnar({"orphan": (("id",), True)})
    with pytest.raises(KeyError, match="does not reference"):
        missing_reference.get_columnar(0)
    missing_reference.close()


def test_hdf5_reader_fixed_only_does_not_access_variable_group(tmp_path):
    """Fixed-only object reads must not require the variable-pool hierarchy."""
    path = tmp_path / "fixed_only.h5"
    particle = RecoParticle(
        id=9,
        index=np.asarray([1, 2, 3], dtype=np.int32),
    )
    data = {
        "index": np.asarray([0]),
        "particles": [ObjectList([particle], RecoParticle())],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(data, cfg={})

    with h5py.File(path, "a") as out_file:
        del out_file["products"]["particles"]["variables"]

    reader = HDF5Reader(
        str(path),
        keys=["particles"],
        build_classes=False,
        fixed_only=True,
    )
    loaded = reader.get(0)["particles"][0]
    assert loaded["id"] == 9
    assert loaded["size"] == 3
    assert "index" not in loaded
    reader.close()


def test_hdf5_reader_fixed_only_rejects_v1_and_mixed_files(tmp_path):
    """Fixed-only loading must not silently degrade on legacy inputs."""
    paths = [tmp_path / "v1.h5", tmp_path / "v2.h5"]
    data = {"index": np.asarray([0]), "value": [np.asarray([1])]}
    for version, path in enumerate(paths, start=1):
        with HDF5Writer(str(path), overwrite=True, format_version=version) as writer:
            writer(data, cfg={})

    with pytest.raises(ValueError, match="only for HDF5 format version 2"):
        HDF5Reader(str(paths[0]), fixed_only=True)
    with pytest.raises(ValueError, match="only for HDF5 format version 2"):
        HDF5Reader([str(path) for path in paths], fixed_only=True)


def test_hdf5_reader_v2_schema_helpers_reject_wrong_types(tmp_path):
    """V2 schema helpers should validate child and attribute types."""
    path = tmp_path / "bad_helpers.h5"
    with h5py.File(path, "w") as out_file:
        out_file.create_group("group")
        out_file.create_dataset("dataset", data=np.asarray([1]))

        with pytest.raises(TypeError, match="HDF5 dataset"):
            require_dataset(out_file, "group")
        with pytest.raises(TypeError, match="HDF5 group"):
            require_group(out_file, "dataset")

    assert decode_string_attribute(b"value", "test") == "value"
    with pytest.raises(TypeError, match="must be a string"):
        decode_string_attribute(3, "test")


def test_hdf5_reader_v2_rejects_bad_product_groups(tmp_path):
    """V2 dispatch should reject missing and unknown product kinds."""
    path = tmp_path / "bad_products.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    reader.keep_open = False
    reader._product_handles = {}

    with h5py.File(path, "w") as out_file:
        out_file.create_dataset("not_group", data=np.asarray([1]))
        unknown = out_file.create_group("unknown")
        unknown.attrs["kind"] = "future"

        with pytest.raises(ValueError, match="not a recognized product group"):
            reader.load_product(out_file, 0, {}, "not_group")
        with pytest.raises(ValueError, match="Unrecognized product kind"):
            reader.load_product(out_file, 0, {}, "unknown")


def test_hdf5_reader_v2_skips_non_group_reconstruction_products(tmp_path):
    """Product reconstruction should ignore physical datasets without metadata."""
    path = tmp_path / "physical_product.h5"
    reader = HDF5Reader.__new__(HDF5Reader)

    with h5py.File(path, "w") as out_file:
        products = out_file.create_group("products")
        products.create_dataset("physical", data=np.asarray([1]))
        data = {"physical": np.asarray([1])}

        reader.reconstruct_products(products, 0, data)

    np.testing.assert_array_equal(data["physical"], [1])


def test_hdf5_reader_v2_rejects_unlinked_product_group(tmp_path):
    """Product loading requires a stable HDF5 path for handle caching."""
    path = tmp_path / "unlinked_product.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    reader.keep_open = False
    reader._product_handles = {}

    with h5py.File(path, "w") as out_file:
        group = out_file.create_group("product")
        group.attrs["kind"] = "array"
        del out_file["product"]

        class UnlinkedContainer:
            """Expose an unlinked group through the minimal container interface."""

            file = out_file

            def __getitem__(self, key):
                return group

        with pytest.raises(ValueError, match="must have an HDF5 path"):
            reader.load_product(UnlinkedContainer(), 0, {}, "product")


@pytest.mark.parametrize("kind", ["array", "objects", "list"])
def test_hdf5_reader_v2_rejects_incomplete_cached_handles(tmp_path, kind):
    """Cached product descriptors must contain every handle their kind needs."""
    path = tmp_path / f"incomplete_{kind}.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    reader.keep_open = True

    with h5py.File(path, "w") as out_file:
        group = out_file.create_group("product")
        group.attrs["kind"] = kind
        cache_key = (str(path), group.name)
        reader._product_handles = {cache_key: _ProductHandles(kind=kind)}

        with pytest.raises(RuntimeError, match="missing|required|Incomplete"):
            reader.load_product(out_file, 0, {}, "product")


def test_hdf5_reader_reconstructs_empty_cluster_particle_table(tmp_path, monkeypatch):
    """V2 reconstruction should retain fields for empty typed particle tables."""
    path = tmp_path / "products.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    particles = spine.data.ObjectList([], spine.data.ParticleLabel())

    with h5py.File(path, "w") as out_file:
        products = out_file.create_group("products")
        group = products.create_group("label")
        group.attrs["product_metadata"] = yaml.safe_dump(
            {
                "product_type": "cluster_label",
                "has_particles": True,
                "has_meta": False,
            }
        )
        monkeypatch.setattr(
            reader,
            "_load_product_child",
            lambda owner, name, entry_idx: particles,
        )
        data = {"label": np.empty((0, 6), dtype=np.float32)}

        reader.reconstruct_products(products, 0, data)

    assert isinstance(data["label"], spine.data.ClusterLabelData)
    assert set(data["label"].particles) == set(spine.data.ParticleLabel().as_dict())


def test_hdf5_reader_reconstruct_product_validation(tmp_path, monkeypatch):
    """V2 reconstruction should reject unknown and untyped empty products."""
    path = tmp_path / "products.h5"
    reader = HDF5Reader.__new__(HDF5Reader)

    with h5py.File(path, "w") as out_file:
        products = out_file.create_group("products")
        objects = products.create_group("objects")
        objects.attrs["product_metadata"] = yaml.safe_dump(
            {"product_type": "object_list", "index_shift_fields": None}
        )
        monkeypatch.setattr(
            reader,
            "_load_product_child",
            lambda owner, name, entry_idx: np.asarray([2]),
        )
        with pytest.raises(ValueError, match="without a stored default class"):
            reader.reconstruct_products(products, 0, {"objects": []})

        data = {"objects": [spine.data.Particle(id=3)]}
        reader.reconstruct_products(products, 0, data)
        assert isinstance(data["objects"], spine.data.ObjectListData)
        assert isinstance(data["objects"].default, spine.data.Particle)
        assert data["objects"].index_shifts == 2

        unknown = products.create_group("unknown")
        unknown.attrs["product_metadata"] = yaml.safe_dump({"product_type": "future"})
        with pytest.raises(ValueError, match="Unknown product type"):
            reader.reconstruct_products(products, 0, {"unknown": np.asarray([1])})


def test_hdf5_reader_requires_owned_product_children(tmp_path):
    """Owned V2 auxiliary payloads should fail clearly when absent."""
    path = tmp_path / "missing_child.h5"
    reader = HDF5Reader.__new__(HDF5Reader)

    with h5py.File(path, "w") as out_file:
        group = out_file.create_group("tensor")
        with pytest.raises(KeyError, match="missing child"):
            reader._load_product_child(group, "meta", 0)


def test_hdf5_reader_v2_rejects_anonymous_object_group(tmp_path):
    """Object products must have a stable path for schema caching."""
    path = tmp_path / "anonymous.h5"
    reader = HDF5Reader.__new__(HDF5Reader)

    with h5py.File(path, "w") as out_file:
        anonymous = out_file.create_group(None)
        with pytest.raises(ValueError, match="must have an HDF5 path"):
            reader.load_product_objects(anonymous, 0, {}, "objects")


@pytest.mark.parametrize("bad_fields", ["index", "[index, 3]"])
def test_hdf5_reader_v2_rejects_bad_variable_pool_fields(tmp_path, bad_fields):
    """Variable-pool field metadata must decode to string lists."""
    path = tmp_path / "bad_fields.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    reader._product_object_schemas = {}
    reader.fixed_only = False

    with h5py.File(path, "w") as out_file:
        objects = out_file.create_group("objects")
        objects.attrs["class_name"] = "RecoParticle"
        objects.attrs["scalar"] = False
        objects.create_dataset(
            "fixed",
            (0,),
            dtype=np.dtype([("_var_offsets_0", np.int64, (2,))]),
        )
        objects.create_dataset("event_offsets", data=np.asarray([0]))
        variables = objects.create_group("variables")
        pool = variables.create_group("pool_0")
        pool.attrs["kind"] = "array"
        pool.attrs["fields"] = bad_fields
        pool.create_dataset("values", data=np.asarray([], dtype=np.int64))

        with pytest.raises(TypeError, match="list of strings"):
            reader.load_product_objects(objects, 0, {}, "objects")


def test_hdf5_reader_v2_rejects_non_group_variable_pool(tmp_path):
    """Every entry below the variables group must itself be a pool group."""
    path = tmp_path / "bad_pool.h5"
    reader = HDF5Reader.__new__(HDF5Reader)
    reader._product_object_schemas = {}
    reader.fixed_only = False

    with h5py.File(path, "w") as out_file:
        objects = out_file.create_group("objects")
        objects.attrs["class_name"] = "RecoParticle"
        objects.attrs["scalar"] = False
        objects.create_dataset("fixed", (0,), dtype=np.dtype([("id", np.int64)]))
        objects.create_dataset("event_offsets", data=np.asarray([0]))
        variables = objects.create_group("variables")
        variables.create_dataset("pool_0", data=np.asarray([], dtype=np.int64))

        with pytest.raises(TypeError, match="must be a group"):
            reader.load_product_objects(objects, 0, {}, "objects")


def test_hdf5_reader_v2_append_and_nested_lists(tmp_path):
    """V2 offsets should remain valid across appends and both jagged layouts."""
    path = tmp_path / "v2_append.h5"

    def batch(base):
        return {
            "index": np.asarray([base, base + 1]),
            "clusters": [
                [np.asarray([1, 2]), np.asarray([3])],
                [np.asarray([], dtype=np.int64)],
            ],
            "features": [
                [
                    np.ones((2, 2), dtype=np.float32),
                    np.ones((3, 3), dtype=np.float32),
                ],
                [
                    np.zeros((1, 2), dtype=np.float32),
                    np.zeros((2, 3), dtype=np.float32),
                ],
            ],
        }

    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(batch(0), cfg={})
    with HDF5Writer(str(path), append=True, format_version=2) as writer:
        writer(batch(2), cfg={})

    reader = HDF5Reader(str(path), build_classes=False)
    assert len(reader) == 4
    assert [array.tolist() for array in reader.get(0)["clusters"]] == [[1, 2], [3]]
    assert [array.shape for array in reader.get(1)["features"]] == [(1, 2), (2, 3)]
    assert [array.tolist() for array in reader.get(2)["clusters"]] == [[1, 2], [3]]
    reader.close()


def test_hdf5_reader_dispatches_mixed_v1_v2_files(tmp_path):
    """One public reader should normalize legacy and offset files identically."""
    paths = [tmp_path / "v1.h5", tmp_path / "v2.h5"]
    data = {
        "index": np.asarray([0]),
        "value": [np.asarray([3, 4, 5], dtype=np.int32)],
        "other": [np.asarray([9], dtype=np.int32)],
    }
    for version, path in enumerate(paths, start=1):
        with HDF5Writer(str(path), overwrite=True, format_version=version) as writer:
            writer(data, cfg={})
    # Files predating explicit layout metadata are legacy V1 by definition.
    with h5py.File(paths[0], "a") as legacy_file:
        del legacy_file["info"].attrs["format_version"]

    reader = HDF5Reader([str(path) for path in paths], keys=["value"])
    assert reader.file_format_versions == [1, 2]
    assert np.array_equal(reader.get(0)["value"], [3, 4, 5])
    assert np.array_equal(reader.get(1)["value"], [3, 4, 5])
    assert "other" not in reader.get(0)
    assert "other" not in reader.get(1)
    reader.close()


def test_hdf5_reader_rejects_unknown_format_version(tmp_path):
    """Unknown physical layouts should fail clearly instead of guessing."""
    path = tmp_path / "future.h5"
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["format_version"] = 99
        info.attrs["version"] = "test"
        info.attrs["cfg"] = "{}"
        info.attrs["complete"] = True
        out_file.create_dataset("events", data=np.empty(0, dtype=np.int64))

    with pytest.raises(ValueError, match="Unsupported HDF5 format version 99"):
        HDF5Reader(str(path))


def test_stage_hdf5_reader_loads_one_stage(tmp_path):
    """Stage cache reader should load products from one named stage."""
    path = tmp_path / "stage_cache.h5"
    source_path = tmp_path / "source.root"
    source_path.write_bytes(b"source-bytes")
    source_stat = source_path.stat()
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray([source_path.name]),
            "source_file_size": np.asarray([source_stat.st_size]),
            "source_file_mtime_ns": np.asarray([source_stat.st_mtime_ns]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
            "file_index": np.asarray([7]),
            "file_entry_index": np.asarray([11]),
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    entry = reader.get(0)
    np.testing.assert_array_equal(entry["dummy_data"], np.asarray([[1.0, 2.0]]))
    assert entry["source_file_entry_index"] == 11
    assert entry["source_file_name"] == source_path.name
    assert entry["source_file_size"] == source_path.stat().st_size
    reader.close()


def test_stage_hdf5_reader_fills_missing_source_entry_index(tmp_path):
    """Older/minimal stage caches should expose source entry metadata."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0, 1]),
            "source_file_name": np.asarray(["source.root", "source.root"]),
            "source_file_size": np.asarray([10, 10]),
            "source_file_mtime_ns": np.asarray([20, 20]),
            "dummy_data": [
                np.asarray([[1.0, 2.0]]),
                np.asarray([[3.0, 4.0]]),
            ],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    entry = reader.get(1)
    assert entry["file_entry_index"] == 1
    assert entry["source_file_entry_index"] == 1
    np.testing.assert_array_equal(entry["dummy_data"], np.asarray([[3.0, 4.0]]))
    reader.close()


def test_stage_hdf5_reader_rejects_incomplete_stages(tmp_path):
    """Incomplete stages should be rejected by default."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.close()

    with pytest.raises(RuntimeError, match="marked incomplete"):
        StageHDF5Reader("deghosting", str(path))


def test_stage_hdf5_reader_can_ignore_incomplete_stages(tmp_path):
    """Incomplete stages can be loaded explicitly when requested."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.close()

    reader = StageHDF5Reader(
        "deghosting", str(path), build_classes=False, ignore_incomplete=True
    )
    assert len(reader) == 1
    reader.close()


def test_stage_hdf5_reader_auto_discovers_unique_product_stage(tmp_path):
    """If no stage is specified, unique product matches should be found automatically."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "fragment_clusts": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("graph_spice")
    writer.close()

    reader = StageHDF5Reader(
        file_keys=str(path), keys=("data_adapt", "fragment_clusts"), build_classes=False
    )
    entry = reader.get(0)
    np.testing.assert_array_equal(entry["data_adapt"], np.asarray([[1.0, 2.0]]))
    np.testing.assert_array_equal(entry["fragment_clusts"], np.asarray([[3.0, 4.0]]))
    reader.close()


def test_stage_hdf5_reader_rejects_ambiguous_product_stage(tmp_path):
    """Auto-discovery should fail if one product exists in multiple stages."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "meta": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "meta": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("graph_spice")
    writer.close()

    with pytest.raises(ValueError, match="appears in multiple stages"):
        StageHDF5Reader(file_keys=str(path), keys=("meta",), build_classes=False)


def test_stage_hdf5_reader_reads_empty_source_info(tmp_path):
    """Missing top-level source provenance should degrade gracefully."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(path, "a") as out_file:
        del out_file["source"]

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    entry = reader.get(0)
    assert "source_file_name" not in entry
    reader.close()


def test_stage_hdf5_reader_decodes_bytes_source_name(tmp_path):
    """Byte-encoded source file names should be decoded on read."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "file_entry_index": np.asarray([7]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(path, "a") as out_file:
        out_file["source"].attrs.modify("file_name", np.bytes_("source.root"))

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    entry = reader.get(0)
    assert entry["source_file_name"] == "source.root"
    assert entry["source_file_entry_index"] == 7
    reader.close()


def test_stage_hdf5_reader_read_source_info_decodes_bytes(tmp_path):
    """Byte-valued source attrs should decode through the helper directly."""
    path = tmp_path / "stage_cache.h5"
    with h5py.File(path, "w") as out_file:
        source = out_file.create_group("source")
        source.attrs["file_name"] = np.bytes_("source.root")
        source.attrs["file_size"] = 10
        source.attrs["file_mtime_ns"] = 20

    with h5py.File(path, "r") as in_file:
        info = StageHDF5Reader.read_source_info(in_file)
    assert info["source_file_name"] == "source.root"


def test_stage_hdf5_reader_rejects_invalid_source_attributes(tmp_path):
    """Source provenance attributes must retain their scalar storage types."""
    path = tmp_path / "stage_cache.h5"
    with h5py.File(path, "w") as out_file:
        source = out_file.create_group("source")
        source.attrs["file_name"] = 3
        source.attrs["file_size"] = 10
        source.attrs["file_mtime_ns"] = 20

    with h5py.File(path, "r") as in_file:
        with pytest.raises(TypeError, match="file_name.*string"):
            StageHDF5Reader.read_source_info(in_file)

    with h5py.File(path, "a") as out_file:
        source = out_file["source"]
        del source.attrs["file_name"]
        del source.attrs["file_size"]
        source.attrs["file_name"] = "source.root"
        source.attrs["file_size"] = 1.5

    with h5py.File(path, "r") as in_file:
        with pytest.raises(TypeError, match="file_size.*scalar integer"):
            StageHDF5Reader.read_source_info(in_file)


def test_stage_hdf5_reader_rejects_non_string_stage_names():
    """The stage-name narrower should reject malformed HDF5 iteration keys."""

    class InvalidStages:
        def __iter__(self):
            return iter([None])

    with pytest.raises(TypeError, match="stage names must be strings"):
        StageHDF5Reader.list_stage_names(InvalidStages())


def test_stage_hdf5_reader_explicit_stage_map_missing_key(tmp_path):
    """Explicit stage maps should fail if the requested product is absent there."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with pytest.raises(KeyError, match="does not exist in stage"):
        StageHDF5Reader(
            file_keys=str(path),
            stage_map={"fragment_clusts": "deghosting"},
            keys=("fragment_clusts",),
            build_classes=False,
        )


def test_stage_hdf5_reader_default_stage_missing_key(tmp_path):
    """Default-stage resolution should fail if the product is absent there."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with pytest.raises(KeyError, match="does not exist in stage"):
        StageHDF5Reader(
            "deghosting",
            str(path),
            keys=("fragment_clusts",),
            build_classes=False,
        )


def test_stage_hdf5_reader_default_stage_resolves_requested_key(tmp_path):
    """Default-stage resolution should accept requested products present there."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    reader = StageHDF5Reader(
        "deghosting", str(path), keys=("data_adapt",), build_classes=False
    )
    np.testing.assert_array_equal(reader.get(0)["data_adapt"], np.asarray([[1.0, 2.0]]))
    reader.close()


def test_stage_hdf5_reader_explicit_stage_map_resolves_key(tmp_path):
    """Explicit stage maps should accept products that exist in that stage."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    reader = StageHDF5Reader(
        file_keys=str(path),
        stage_map={"data_adapt": "deghosting"},
        keys=("data_adapt",),
        build_classes=False,
    )
    np.testing.assert_array_equal(reader.get(0)["data_adapt"], np.asarray([[1.0, 2.0]]))
    reader.close()


def test_stage_hdf5_reader_rejects_missing_product(tmp_path):
    """Automatic discovery should fail when no stage contains the product."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "data_adapt": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with pytest.raises(KeyError, match="Could not find requested product"):
        StageHDF5Reader(
            file_keys=str(path), keys=("fragment_clusts",), build_classes=False
        )


def test_stage_hdf5_reader_validate_stage_lengths_edges():
    """Stage-length validation should handle empty and mismatched inputs."""
    assert StageHDF5Reader.validate_stage_lengths("dummy.h5", {}) == 0
    with pytest.raises(ValueError, match="do not expose the same number of entries"):
        StageHDF5Reader.validate_stage_lengths(
            "dummy.h5", {"deghosting": 1, "graph_spice": 2}
        )


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (12, "attribute must be a string"),
        ("12", "decode to a mapping"),
        ("1: value", "keys must be strings"),
    ],
)
def test_stage_hdf5_reader_process_cfg_rejects_invalid_payload(tmp_path, cfg, message):
    """Malformed stage configuration payloads should fail explicitly."""
    path = tmp_path / "stage_cache.h5"
    with h5py.File(path, "w") as out_file:
        out_file.create_group("info").attrs["version"] = "test"
        stages = out_file.create_group("stages")
        stage = stages.create_group("deghosting")
        info = stage.create_group("info")
        info.attrs["complete"] = True
        info.attrs["cfg"] = cfg
        stage.create_dataset("dummy_data", data=np.asarray([[1.0, 2.0]]))
        ref_dtype = np.dtype([("dummy_data", h5py.regionref_dtype)])
        events = stage.create_dataset("events", shape=(1,), dtype=ref_dtype)
        events[0] = (stage["dummy_data"].regionref[:],)

    with pytest.raises(TypeError, match=message):
        StageHDF5Reader("deghosting", str(path), build_classes=False)


def test_stage_hdf5_reader_process_cfg_parser_error_returns_none(monkeypatch, tmp_path):
    """Malformed stage cfg payloads should warn and produce None."""
    path = tmp_path / "stage_cache.h5"
    with h5py.File(path, "w") as out_file:
        out_file.create_group("info").attrs["version"] = "test"
        stages = out_file.create_group("stages")
        stage = stages.create_group("deghosting")
        info = stage.create_group("info")
        info.attrs["complete"] = True
        info.attrs["cfg"] = "{}"
        stage.create_dataset("dummy_data", data=np.asarray([[1.0, 2.0]]))
        ref_dtype = np.dtype([("dummy_data", h5py.regionref_dtype)])
        events = stage.create_dataset("events", shape=(1,), dtype=ref_dtype)
        events[0] = (stage["dummy_data"].regionref[:],)

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    assert reader.cfg == {}
    reader.close()

    monkeypatch.setattr(
        "spine.io.read.stage_hdf5.yaml.safe_load",
        lambda _: (_ for _ in ()).throw(ParserError(None, None, None, None)),
    )

    with pytest.warns(UserWarning, match="Parsing stage configuration failed"):
        reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    assert reader.cfg is None
    reader.close()


def test_stage_hdf5_reader_rejects_bad_index_and_unnamed_events(tmp_path):
    """Stage reader should reject out-of-range entries and malformed events."""
    path = tmp_path / "stage_cache.h5"
    with h5py.File(path, "w") as out_file:
        out_file.create_group("info").attrs["version"] = "test"
        stages = out_file.create_group("stages")
        stage = stages.create_group("deghosting")
        info = stage.create_group("info")
        info.attrs["complete"] = True
        stage.create_dataset("dummy_data", data=np.asarray([[1.0, 2.0]]))
        stage.create_dataset("events", data=np.asarray([0], dtype=np.int64))

    reader = StageHDF5Reader("deghosting", str(path), build_classes=False)
    with pytest.raises(IndexError, match="out of bounds"):
        reader.get(1)
    with pytest.raises(ValueError, match="does not have named fields"):
        reader.get(0)
    reader.close()


def test_stage_hdf5_reader_closes_ephemeral_handle(tmp_path):
    """Single-access staged reads should close temporary handles when requested."""
    path = tmp_path / "stage_cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    reader = StageHDF5Reader(
        "deghosting", str(path), build_classes=False, keep_open=False
    )
    entry = reader.get(0)
    np.testing.assert_array_equal(entry["dummy_data"], np.asarray([[1.0, 2.0]]))


def test_resolve_legacy_meta_class_2d():
    """Test legacy Meta class names map to explicit 2D metadata classes."""
    array = np.array(
        [
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float32),
                np.array([2, 2], dtype=np.int64),
            )
        ],
        dtype=[
            ("lower", np.float32, (2,)),
            ("upper", np.float32, (2,)),
            ("size", np.float32, (2,)),
            ("count", np.int64, (2,)),
        ],
    )

    assert HDF5Reader.resolve_object_class("Meta", array) is ImageMeta2D


def test_resolve_explicit_meta_class_2d():
    """Explicit ImageMeta2D class names should resolve without spine.data export."""
    assert spine.data.ImageMeta2D is ImageMeta2D
    array = np.empty(0, dtype=[("count", np.int64, (2,))])
    assert HDF5Reader.resolve_object_class("ImageMeta2D", array) is ImageMeta2D


def test_resolve_legacy_meta_class_3d():
    """Test legacy Meta class names map to explicit 3D metadata classes."""
    array = np.array(
        [
            (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.array([0.5, 0.5, 0.5], dtype=np.float32),
                np.array([2, 2, 2], dtype=np.int64),
            )
        ],
        dtype=[
            ("lower", np.float32, (3,)),
            ("upper", np.float32, (3,)),
            ("size", np.float32, (3,)),
            ("count", np.int64, (3,)),
        ],
    )

    assert HDF5Reader.resolve_object_class("Meta", array) is ImageMeta3D


def test_resolve_explicit_meta_class_3d():
    """Explicit ImageMeta3D class names should resolve without spine.data export."""
    assert spine.data.ImageMeta3D is ImageMeta3D
    array = np.empty(0, dtype=[("count", np.int64, (3,))])
    assert HDF5Reader.resolve_object_class("ImageMeta3D", array) is ImageMeta3D


def test_resolve_object_class_errors():
    """Legacy Meta resolution should reject malformed metadata."""
    bad = np.array([(1,)], dtype=[("x", np.int64)])
    with pytest.raises(TypeError, match="requires a structured dtype"):
        HDF5Reader.resolve_object_class("Meta", bad)

    bad_dim = np.array(
        [(np.array([1], dtype=np.int64),)],
        dtype=[("count", np.int64, (1,))],
    )
    with pytest.raises(ValueError, match="Unsupported legacy Meta dimensionality"):
        HDF5Reader.resolve_object_class("Meta", bad_dim)


def test_resolve_legacy_meta_class_empty_array_defaults_3d():
    """Empty legacy metadata arrays should fall back to 3D metadata."""
    array = np.empty(0, dtype=[("count", np.int64, (3,))])
    assert HDF5Reader.resolve_object_class("Meta", array) is ImageMeta3D


def test_process_cfg_parser_error_returns_none(monkeypatch, hdf5_data):
    """Malformed legacy configuration payloads should warn and return None."""
    monkeypatch.setattr(
        "spine.io.read.hdf5.reader.yaml.safe_load",
        lambda _: (_ for _ in ()).throw(ParserError(None, None, None, None)),
    )

    with pytest.warns(UserWarning, match="Parsing configuration failed"):
        reader = HDF5Reader(hdf5_data, build_classes=False)

    assert reader.cfg is None
    reader.close()


def test_get_rejects_events_without_named_fields(tmp_path):
    """Event entries must expose named fields for key loading."""
    path = tmp_path / "bad_events.h5"
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["version"] = "test"
        info.attrs["cfg"] = "{}"
        out_file.create_dataset("events", data=np.asarray([0], dtype=np.int64))

    reader = HDF5Reader(str(path), build_classes=False)

    with pytest.raises(ValueError, match="does not have named fields"):
        reader.get(0)

    reader.close()


def test_load_key_object_dataset_builds_and_filters_unknown_attrs(tmp_path):
    """Structured object datasets should support filtering unknown attrs and raw dict output."""
    path = tmp_path / "objects.h5"
    dtype = np.dtype(
        [
            ("run", np.int64),
            ("subrun", np.int64),
            ("event", np.int64),
            ("extra", np.int64),
        ]
    )

    with h5py.File(path, "w") as out_file:
        dataset = out_file.create_dataset(
            "run_info", data=np.asarray([(1, 2, 3, 9)], dtype=dtype)
        )
        dataset.attrs["class_name"] = "RunInfo"
        dataset.attrs["scalar"] = True

        reader = HDF5Reader.__new__(HDF5Reader)
        reader.skip_unknown_attrs = True
        reader.build_classes = True
        built = {}
        reader.load_region_product(
            out_file, {"run_info": np.s_[0:1]}, built, "run_info"
        )
        assert built["run_info"].run == 1

        reader.build_classes = False
        raw = {}
        reader.load_region_product(out_file, {"run_info": np.s_[0:1]}, raw, "run_info")
        assert raw["run_info"] == {"run": 1, "subrun": 2, "event": 3}


def test_load_key_rejects_non_string_object_class(tmp_path):
    """Legacy object datasets require a string reconstruction class name."""
    path = tmp_path / "bad_object_class.h5"
    dtype = np.dtype([("id", np.int64)])

    with h5py.File(path, "w") as out_file:
        dataset = out_file.create_dataset(
            "objects", data=np.asarray([(1,)], dtype=dtype)
        )
        dataset.attrs["class_name"] = 3
        dataset.attrs["scalar"] = False
        event = {"objects": dataset.regionref[0:1]}
        reader = HDF5Reader.__new__(HDF5Reader)

        with pytest.raises(TypeError, match="string 'class_name'"):
            reader.load_region_product(out_file, event, {}, "objects")


def test_load_key_group_paths(tmp_path):
    """Grouped datasets should support both shared-element and per-element storage."""
    path = tmp_path / "groups.h5"

    with h5py.File(path, "w") as out_file:
        shared = out_file.create_group("shared")
        elements = shared.create_dataset(
            "elements", data=np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        )
        shared_index = shared.create_dataset("index", (1,), dtype=h5py.regionref_dtype)
        shared_index[0] = elements.regionref[0:2]

        split = out_file.create_group("split")
        element_0 = split.create_dataset(
            "element_0", data=np.asarray([[1, 2]], dtype=np.int64)
        )
        element_1 = split.create_dataset(
            "element_1", data=np.asarray([[3, 4, 5]], dtype=np.int64)
        )
        split_index = split.create_dataset("index", (1, 2), dtype=h5py.regionref_dtype)
        split_index[0, 0] = element_0.regionref[0:1]
        split_index[0, 1] = element_1.regionref[0:1]

        reader = HDF5Reader.__new__(HDF5Reader)
        data = {}
        reader.load_region_product(out_file, {"shared": np.s_[0:1]}, data, "shared")
        reader.load_region_product(out_file, {"split": np.s_[0:1]}, data, "split")

        assert len(data["shared"]) == 1
        assert data["shared"][0].shape == (2, 2)
        assert len(data["split"]) == 2
        assert data["split"][1].shape == (1, 3)


def test_load_key_rejects_unknown_storage_kind():
    """Loading should reject objects that are neither datasets nor groups."""

    class DummyFile(dict):
        pass

    reader = HDF5Reader.__new__(HDF5Reader)
    data = {}
    with pytest.raises(ValueError, match="neither an HDF5 group nor dataset"):
        reader.load_region_product(
            DummyFile(bad=object()), {"bad": slice(None)}, data, "bad"
        )
