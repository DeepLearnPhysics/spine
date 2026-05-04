"""Tests for the HDF5 reader."""

import multiprocessing

import h5py
import numpy as np
import pytest
from yaml.parser import ParserError

import spine.data
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.read import HDF5Reader


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

    monkeypatch.setattr("spine.io.read.hdf5.h5py.File", counted_file)
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
    monkeypatch.setattr("spine.io.read.hdf5.h5py.File", counted_file)
    monkeypatch.setattr("spine.io.read.hdf5.os.getpid", lambda: next(pids))

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

    monkeypatch.setattr("spine.io.read.hdf5.h5py.File", counted_file)
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
    with pytest.raises(AssertionError, match="requires a structured dtype"):
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
        "spine.io.read.hdf5.yaml.safe_load",
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
        reader.load_key(out_file, {"run_info": np.s_[0:1]}, built, "run_info")
        assert built["run_info"].run == 1

        reader.build_classes = False
        raw = {}
        reader.load_key(out_file, {"run_info": np.s_[0:1]}, raw, "run_info")
        assert raw["run_info"] == {"run": 1, "subrun": 2, "event": 3}


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
        reader.load_key(out_file, {"shared": np.s_[0:1]}, data, "shared")
        reader.load_key(out_file, {"split": np.s_[0:1]}, data, "split")

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
    with pytest.raises(ValueError, match="neither a group nor dataset"):
        reader.load_key(DummyFile(bad=object()), {"bad": slice(None)}, data, "bad")
