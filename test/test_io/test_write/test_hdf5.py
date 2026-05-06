"""Test that the writer classes work as intended."""

import os

import h5py
import numpy as np
import pytest

from spine.data import (
    CRTHit,
    Flash,
    Meta,
    Neutrino,
    ObjectList,
    Particle,
    RecoParticle,
    RunInfo,
    Trigger,
)
from spine.io.write import *


@pytest.fixture(name="hdf5_output")
def fixture_hdf5_output(tmp_path):
    """Create a dummy output path for an HDF5 file.

    Parameters
    ----------
    tmp_path : str
       Generic pytest fixture used to handle temporary test files
    """
    return os.path.join(tmp_path, "dummy.h5")


@pytest.fixture(name="tensor_list")
def fixture_tensor_list(request):
    """Generates a dummy list of unwrapped tensors."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of tensors of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    tensors = []
    for i, s in enumerate(sizes):
        tensors.append(np.random.rand(s, 5))

    return tensors


@pytest.fixture(name="index_list")
def fixture_index_list(request):
    """Generates a dummy list of unwrapped index lists."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    indexes = []
    default = np.empty(0, dtype=np.int64)
    for i, s in enumerate(sizes):
        index = np.arange(s)
        if s > 1:
            index = np.split(index, [np.random.randint(1, s)])
            indexes.append(index)
        elif s == 1:
            indexes.append([index])
        else:
            indexes.append(ObjectList(index, default))

    return indexes


@pytest.fixture(name="edge_index_list")
def fixture_edge_index_list(request):
    """Generates a dummy list of unwrapped edge indexes."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    edge_indexes = []
    for i, s in enumerate(sizes):
        edge_indexes.append(np.random.rand(s, 2))

    return edge_indexes


@pytest.mark.parametrize(
    "tensor_list, index_list, edge_index_list",
    [(0, 0, 0), (10, 10, 25), ((0, 0), (0, 0), (0, 0)), ((5, 10), (5, 10), (8, 25))],
    indirect=True,
)
def test_hdf5_writer(hdf5_output, tensor_list, index_list, edge_index_list):
    """Tests the HDF5 writer."""
    # Create an output similar to that of the full chain
    batch_size = len(tensor_list)
    sizes = [len(t) for t in tensor_list]
    data = {
        "index": np.arange(batch_size),
        "dummy_data": tensor_list,
        "dummy_meta": [Meta()] * batch_size,
        "dummy_run_info": [RunInfo()] * batch_size,
        "dummy_trigger": [Trigger()] * batch_size,
        "dummy_particles": generate_object_list(Particle, sizes),
        "dummy_neutrinos": generate_object_list(Neutrino, sizes),
        "dummy_flashes": generate_object_list(Flash, sizes),
        "dummy_crthits": generate_object_list(CRTHit, sizes),
        "dummy_tensor": tensor_list,
        "dummy_clusts": index_list,
        "dummy_edge_index": edge_index_list,
    }

    # Initialize the writer
    writer = HDF5Writer(hdf5_output)

    # Write output
    writer(data)


def test_hdf5_writer_file_name_inferred_from_prefix():
    """Test HDF5 output file name inference from input prefixes."""
    assert HDF5Writer.get_file_names(None, "input", split=False) == ["input_spine.h5"]
    assert HDF5Writer.get_file_names(None, ["a", "b"], split=True) == [
        "a_spine.h5",
        "b_spine.h5",
    ]


def test_hdf5_writer_file_name_inferred_from_prefix_with_directory(tmp_path):
    """A writer directory should relocate inferred names without changing stems."""
    output_dir = tmp_path / "outputs"
    assert HDF5Writer.get_file_names(
        None, "input.root", split=False, directory=str(output_dir), suffix="cache"
    ) == [os.path.join(output_dir, "input_cache.h5")]
    assert HDF5Writer.get_file_names(
        None,
        ["a.root", "b.root"],
        split=True,
        directory=str(output_dir),
        suffix="cache",
    ) == [
        os.path.join(output_dir, "a_cache.h5"),
        os.path.join(output_dir, "b_cache.h5"),
    ]


def test_hdf5_writer_explicit_file_name_with_directory(tmp_path):
    """A writer directory should relocate explicit output names too."""
    output_dir = tmp_path / "outputs"
    file_name = os.path.join(tmp_path, "custom.h5")
    assert HDF5Writer.get_file_names(
        file_name, split=False, directory=str(output_dir)
    ) == [os.path.join(output_dir, "custom.h5")]


def test_hdf5_writer_split_explicit_single_file(hdf5_output):
    """Test split output keeps an explicit name when there is one input file."""
    assert HDF5Writer.get_file_names(hdf5_output, ["input"], split=True) == [
        hdf5_output
    ]


def test_hdf5_writer_split_explicit_multiple_files(tmp_path):
    """Test split output enumerates an explicit name for multiple input files."""
    file_name = os.path.join(tmp_path, "output.h5")

    assert HDF5Writer.get_file_names(file_name, ["a", "b", "c"], split=True) == [
        os.path.join(tmp_path, "output_0.h5"),
        os.path.join(tmp_path, "output_1.h5"),
        os.path.join(tmp_path, "output_2.h5"),
    ]


def test_hdf5_writer_split_explicit_multiple_files_with_directory(tmp_path):
    """Split explicit names should relocate into the requested directory."""
    file_name = os.path.join(tmp_path, "output.h5")
    output_dir = tmp_path / "outputs"

    assert HDF5Writer.get_file_names(
        file_name, ["a", "b", "c"], split=True, directory=str(output_dir)
    ) == [
        os.path.join(output_dir, "output_0.h5"),
        os.path.join(output_dir, "output_1.h5"),
        os.path.join(output_dir, "output_2.h5"),
    ]


def test_hdf5_writer_append_existing_file(hdf5_output):
    """Test appending a batch to an existing HDF5 output file."""
    data = {
        "index": np.arange(2),
        "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
    }

    HDF5Writer(hdf5_output)(data)
    HDF5Writer(hdf5_output, append=True)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        assert len(out_file["events"]) == 4


def test_hdf5_writer_append_missing_file(hdf5_output):
    """Test append mode creates a missing HDF5 output file."""
    data = {
        "index": np.arange(2),
        "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
    }

    HDF5Writer(hdf5_output, append=True)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        assert len(out_file["events"]) == 2


def test_hdf5_writer_auto_stores_source_entry_provenance(hdf5_output):
    """Writer should persist original source entry provenance automatically."""
    writer = HDF5Writer(
        hdf5_output,
        overwrite=True,
        keys=["dummy_data"],
    )
    writer(
        {
            "index": np.asarray([0, 1]),
            "file_index": np.asarray([3, 3]),
            "file_entry_index": np.asarray([11, 12]),
            "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "source_file_index" in out_file
        assert "source_file_entry_index" in out_file
        np.testing.assert_array_equal(out_file["source_file_index"][:], [3, 3])
        np.testing.assert_array_equal(out_file["source_file_entry_index"][:], [11, 12])


def test_hdf5_writer_skip_keys_can_disable_source_provenance(hdf5_output):
    """Explicit skip keys should still be able to drop source provenance."""
    writer = HDF5Writer(
        hdf5_output,
        overwrite=True,
        skip_keys=["source_file_index", "source_file_entry_index"],
    )
    writer(
        {
            "index": np.asarray([0]),
            "file_index": np.asarray([2]),
            "file_entry_index": np.asarray([9]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "source_file_index" not in out_file
        assert "source_file_entry_index" not in out_file


def test_hdf5_writer_finalize_marks_output_complete(hdf5_output):
    """Finalize should mark written files as complete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_close_leaves_unfinalized_output_incomplete(hdf5_output):
    """Closing alone should not mark files as complete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_stage_hdf5_writer_finalizes_stages_independently(hdf5_output):
    """Each cache stage should track completeness independently."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
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
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        assert not out_file["stages"]["graph_spice"]["info"].attrs["complete"]
        assert len(out_file["stages"]["deghosting"]["events"]) == 1
        assert len(out_file["stages"]["graph_spice"]["events"]) == 1


def test_stage_hdf5_writer_overwrite_stage_preserves_other_stages(hdf5_output):
    """Rewriting one stage should leave sibling stages untouched."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
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
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("graph_spice")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[5.0, 6.0]])],
        },
        overwrite_stage=True,
    )
    writer.finalize_stage("graph_spice")
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        np.testing.assert_array_equal(
            out_file["stages"]["deghosting"]["dummy_data"][:],
            np.asarray([[1.0, 2.0]]),
        )
        np.testing.assert_array_equal(
            out_file["stages"]["graph_spice"]["dummy_data"][:],
            np.asarray([[5.0, 6.0]]),
        )


def test_stage_hdf5_writer_lists_written_stages(hdf5_output):
    """Stage cache writer should expose the written stage names."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
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
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    assert writer.list_stages() == ("deghosting", "graph_spice")
    writer.close()


def test_stage_hdf5_writer_stores_source_provenance(tmp_path):
    """Stage cache files should persist lightweight source-file provenance."""
    source_path = tmp_path / "source.root"
    source_path.write_bytes(b"source-bytes")
    cache_path = tmp_path / "cache.h5"

    stat_result = source_path.stat()
    writer = StageHDF5Writer(str(cache_path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray([source_path.name]),
            "source_file_size": np.asarray([stat_result.st_size]),
            "source_file_mtime_ns": np.asarray([stat_result.st_mtime_ns]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(cache_path, "r") as out_file:
        source_group = out_file["source"]
        assert source_group.attrs["file_name"] == source_path.name
        assert source_group.attrs["file_size"] == stat_result.st_size
        assert source_group.attrs["file_mtime_ns"] == stat_result.st_mtime_ns


def test_stage_hdf5_writer_rejects_mismatched_source(tmp_path):
    """Writing a later stage with different source provenance should fail."""
    source_a = tmp_path / "source_a.root"
    source_b = tmp_path / "source_b.root"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"bb")
    cache_path = tmp_path / "cache.h5"

    stat_a = source_a.stat()
    stat_b = source_b.stat()
    writer = StageHDF5Writer(str(cache_path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray([source_a.name]),
            "source_file_size": np.asarray([stat_a.st_size]),
            "source_file_mtime_ns": np.asarray([stat_a.st_mtime_ns]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    writer = StageHDF5Writer(str(cache_path))
    with pytest.raises(RuntimeError, match="Cache source mismatch"):
        writer.write_stage(
            "graph_spice",
            {
                "index": np.asarray([0]),
                "source_file_name": np.asarray([source_b.name]),
                "source_file_size": np.asarray([stat_b.st_size]),
                "source_file_mtime_ns": np.asarray([stat_b.st_mtime_ns]),
                "dummy_data": [np.asarray([[3.0, 4.0]])],
            },
        )
    writer.close()


def test_stage_hdf5_writer_splits_batch_by_source(tmp_path):
    """A mixed-source batch should be routed into one cache file per source."""
    output = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(output), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0, 1]),
            "source_file_name": np.asarray(["source_a.root", "source_b.root"]),
            "source_file_size": np.asarray([10, 20]),
            "source_file_mtime_ns": np.asarray([30, 40]),
            "source_file_entry_index": np.asarray([5, 6]),
            "dummy_data": [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    source_a_cache = tmp_path / "source_a_stage.h5"
    source_b_cache = tmp_path / "source_b_stage.h5"
    assert source_a_cache.is_file()
    assert source_b_cache.is_file()

    with h5py.File(source_a_cache, "r") as out_file:
        assert out_file["source"].attrs["file_name"] == "source_a.root"
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        np.testing.assert_array_equal(
            out_file["stages"]["deghosting"]["source_file_entry_index"][:], [5]
        )

    with h5py.File(source_b_cache, "r") as out_file:
        assert out_file["source"].attrs["file_name"] == "source_b.root"
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        np.testing.assert_array_equal(
            out_file["stages"]["deghosting"]["source_file_entry_index"][:], [6]
        )


def test_stage_hdf5_writer_overwrite_removes_existing_output(hdf5_output):
    """Overwrite should remove an existing target cache file eagerly."""
    open(hdf5_output, "a", encoding="utf-8").close()
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    assert not os.path.exists(hdf5_output)
    writer.close()


def test_stage_hdf5_writer_close_swallows_handle_errors(hdf5_output):
    """Writer cleanup should clear state even if one handle raises on close."""
    writer = StageHDF5Writer(hdf5_output)

    class BadHandle:
        def close(self):
            raise OSError("boom")

    writer._handles["x"] = BadHandle()
    writer._handle_pid = 123
    writer.close()
    assert writer._handles == {}
    assert writer._handle_pid is None


def test_stage_hdf5_writer_rejects_pid_change(hdf5_output, monkeypatch):
    """Persistent staged-writer handles should stay process-local."""
    writer = StageHDF5Writer(hdf5_output)
    writer._handle_pid = 1
    monkeypatch.setattr("spine.io.write.stage_hdf5.os.getpid", lambda: 2)
    with pytest.raises(RuntimeError, match="process-local"):
        writer._check_handle_pid()


def test_stage_hdf5_writer_open_handle_keep_open_false(hdf5_output):
    """Disabling persistent handles should return a close-on-use handle."""
    writer = StageHDF5Writer(hdf5_output, keep_open=False, overwrite=True)
    handle, should_close = writer._open_handle(hdf5_output)
    assert should_close is True
    handle.close()
    writer.close()


def test_stage_hdf5_writer_open_handle_reopens_invalid_cached_handle(hdf5_output):
    """Persistent handle lookup should reopen invalid cached handles."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    handle, _ = writer._open_handle(hdf5_output)
    handle.close()
    reopened, should_close = writer._open_handle(hdf5_output)
    assert reopened.id.valid
    assert should_close is False
    writer.close()


def test_stage_hdf5_writer_get_batch_source_info_edges(hdf5_output):
    """Source provenance extraction should cover missing, scalar, and bad inputs."""
    writer = StageHDF5Writer(hdf5_output)
    with pytest.raises(KeyError, match="Missing keys"):
        writer.get_batch_source_info({"index": np.asarray([0])})

    info = writer.get_batch_source_info(
        {
            "source_file_name": np.asarray("source.root"),
            "source_file_size": np.asarray(10),
            "source_file_mtime_ns": np.asarray(20),
        }
    )
    assert info == {"file_name": "source.root", "file_size": 10, "file_mtime_ns": 20}

    info = writer.get_batch_source_info(
        {
            "source_file_name": "source.root",
            "source_file_size": 10,
            "source_file_mtime_ns": 20,
        }
    )
    assert info["file_size"] == 10

    with pytest.raises(ValueError, match="is empty"):
        writer.get_batch_source_info(
            {
                "source_file_name": np.asarray([], dtype=object),
                "source_file_size": np.asarray([10]),
                "source_file_mtime_ns": np.asarray([20]),
            }
        )

    with pytest.raises(ValueError, match="contains multiple values"):
        writer.get_batch_source_info(
            {
                "source_file_name": np.asarray(["a.root", "b.root"]),
                "source_file_size": np.asarray([10, 10]),
                "source_file_mtime_ns": np.asarray([20, 20]),
            }
        )


def test_stage_hdf5_writer_ensure_source_group_existing_matches_and_mismatches(
    tmp_path,
):
    """Existing source groups should validate cache-file provenance."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer._ensure_file(str(path))
    handle, should_close = writer._open_handle(str(path))
    try:
        batch = {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
        }
        writer.ensure_source_group(handle, batch, str(path))
        writer.ensure_source_group(handle, batch, str(path))
        bad = dict(batch)
        bad["source_file_size"] = np.asarray([11])
        with pytest.raises(RuntimeError, match="Cache source mismatch"):
            writer.ensure_source_group(handle, bad, str(path))
    finally:
        if should_close:
            handle.close()
        writer.close()


def test_stage_hdf5_writer_prepare_batch_scalar_and_skip_keys(tmp_path):
    """Single-entry normalization and skip-key filtering should both work."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    batch, batch_size = writer._prepare_batch(
        {
            "index": np.int64(0),
            "file_index": np.int64(1),
            "file_entry_index": np.int64(2),
            "source_file_name": "source.root",
            "source_file_size": 10,
            "source_file_mtime_ns": 20,
            "dummy_data": np.asarray([[1.0, 2.0]]),
        }
    )
    assert batch_size == 1
    assert isinstance(batch["index"], list)
    state = StageHDF5Writer(str(path), overwrite=True)
    state.skip_keys = ["dummy_data"]
    stage_state = state._create_stage_state("deghosting", batch)
    assert "dummy_data" not in stage_state.keys
    assert "source_file_name" not in stage_state.keys
    state.close()
    writer.close()


def test_stage_hdf5_writer_output_path_and_split_missing_keys(tmp_path):
    """Output-path resolution and split validation should cover edge paths."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    assert writer.get_output_path({"file_name": "source.root"}) == str(path)
    assert writer.get_output_path({"file_name": "source.root"}, True).endswith(
        "source_stage.h5"
    )
    with pytest.raises(KeyError, match="Missing key"):
        writer.split_batch_by_source({"index": np.asarray([0])})
    writer.close()


def test_stage_hdf5_writer_uses_explicit_directory(tmp_path):
    """Stage cache output paths should honor an explicit output directory."""
    directory = tmp_path / "cache_dir"
    writer = StageHDF5Writer(
        str(tmp_path / "cache.h5"), overwrite=True, directory=str(directory)
    )
    assert writer.get_output_path({"file_name": "source.root"}) == os.path.join(
        directory, "cache.h5"
    )
    assert writer.get_output_path({"file_name": "source.root"}, True) == os.path.join(
        directory, "source_stage.h5"
    )
    writer.close()


def test_stage_hdf5_writer_accepts_prefix_with_directory(tmp_path):
    """Stage cache naming should support prefix-based defaults under a new directory."""
    directory = tmp_path / "cache_dir"
    writer = StageHDF5Writer(
        file_name=None,
        prefix="input.root",
        overwrite=True,
        directory=str(directory),
        suffix="cache",
    )
    assert writer.file_name == os.path.join(directory, "input_cache.h5")
    assert writer.get_output_path({"file_name": "source.root"}) == os.path.join(
        directory, "input_cache.h5"
    )
    assert writer.get_output_path({"file_name": "source.root"}, True) == os.path.join(
        directory, "source_cache.h5"
    )
    writer.close()


def test_stage_hdf5_writer_requires_split_mode(tmp_path):
    """Stage caches should reject non-split writer configuration."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"))
    assert writer.split is True
    writer.close()

    with pytest.raises(ValueError, match="split=True"):
        StageHDF5Writer(str(tmp_path / "cache.h5"), split=False)


def test_stage_hdf5_writer_split_batch_preserves_scalar_values(tmp_path):
    """Source splitting should preserve scalar-valued batch metadata."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"), overwrite=True)
    groups = writer.split_batch_by_source(
        {
            "index": np.asarray([0, 1]),
            "epoch": 3,
            "source_file_name": np.asarray(["a.root", "b.root"]),
            "source_file_size": np.asarray([10, 20]),
            "source_file_mtime_ns": np.asarray([30, 40]),
            "dummy_data": [np.asarray([[1.0]]), np.asarray([[2.0]])],
        }
    )
    assert groups[0][1]["epoch"] == 3
    assert groups[1][1]["epoch"] == 3
    writer.close()


def test_stage_hdf5_writer_stage_group_existing_paths_and_flush(tmp_path):
    """Existing-stage update branches and flush bookkeeping should be covered."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True, flush_frequency=1)
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer.write_stage("deghosting", batch, cfg={"a": 1}, attrs={"tag": "x"})
    writer.write_stage("deghosting", batch, cfg={"a": 2}, attrs={"tag": "y"})
    writer.close()

    with h5py.File(path, "r") as out_file:
        info = out_file["stages"]["deghosting"]["info"].attrs
        assert info["tag"] == "y"
        assert "cfg" in info


def test_stage_hdf5_writer_rejects_reopen_existing_stage_across_sessions(tmp_path):
    """Appending to an existing stage from a new writer session should fail."""
    path = tmp_path / "cache.h5"
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage("deghosting", batch)
    writer.close()

    writer = StageHDF5Writer(str(path))
    normalized, _ = writer._prepare_batch(batch)
    state = writer._create_stage_state("deghosting", normalized)
    out_file, should_close = writer._open_handle(str(path))
    try:
        del writer._stage_states["deghosting"]
        with pytest.raises(
            RuntimeError, match="Reopening and appending an existing stage"
        ):
            writer._ensure_stage_group(out_file, str(path), "deghosting", state)
    finally:
        if should_close:
            out_file.close()
    writer.close()


def test_stage_hdf5_writer_keep_open_false_closes_in_write_finalize_and_list(tmp_path):
    """Close-on-use mode should exercise all staged-writer close branches."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True, keep_open=False)
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer.write_stage("deghosting", batch)
    writer.finalize_stage("deghosting")
    assert writer.list_stages() == ("deghosting",)
    writer.close()


def test_stage_hdf5_writer_finalize_and_list_ignore_missing_stage(tmp_path):
    """Finalize/list helpers should tolerate files without the requested stage."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"), overwrite=True)
    batch = {
        "index": np.asarray([0, 1]),
        "source_file_name": np.asarray(["a.root", "b.root"]),
        "source_file_size": np.asarray([10, 20]),
        "source_file_mtime_ns": np.asarray([30, 40]),
        "dummy_data": [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])],
    }
    writer.write_stage("deghosting", batch)
    writer.finalize_stage("graph_spice")
    assert writer.list_stages() == ("deghosting",)
    writer.close()


def test_hdf5_writer_stores_stored_properties(hdf5_output):
    """Test HDF5 writer serializes stored properties on data objects."""
    particle = RecoParticle(id=3, index=np.arange(4, dtype=np.int64), pid=2)
    data = {
        "index": np.array([0]),
        "particles": [ObjectList([particle], RecoParticle())],
    }

    HDF5Writer(hdf5_output)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        fields = out_file["particles"].dtype.names
        assert "size" in fields
        assert "pdg_code" in fields
        assert "mass" in fields
        assert "ke" in fields
        assert "momentum" in fields
        assert "p" in fields
        assert "reco_ke" not in fields
        assert out_file["particles"]["size"][0] == 4
        assert out_file["particles"]["pdg_code"][0] == 13


def test_hdf5_writer_rejects_existing_file(hdf5_output):
    """Writer should protect existing outputs unless overwrite/append is enabled."""
    open(hdf5_output, "a", encoding="utf-8").close()
    with pytest.raises(FileExistsError):
        HDF5Writer(hdf5_output)


def test_hdf5_writer_get_file_names_errors():
    """File name inference should reject incompatible inputs."""
    with pytest.raises(AssertionError, match="must provide the input file `prefix`"):
        HDF5Writer.get_file_names(None, None, split=False)
    with pytest.raises(
        AssertionError, match="must provide one `prefix` per input file"
    ):
        HDF5Writer.get_file_names("out.h5", "prefix", split=True)


def test_hdf5_writer_get_stored_keys_errors(hdf5_output):
    """Stored key selection should reject inconsistent requests."""
    writer = HDF5Writer(hdf5_output, keys=["a"], skip_keys=["b"])
    with pytest.raises(AssertionError, match="Must not specify both"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})

    writer = HDF5Writer(hdf5_output, skip_keys=["missing"])
    with pytest.raises(KeyError, match="appears in `skip_keys`"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})

    writer = HDF5Writer(hdf5_output, keys=["missing"])
    with pytest.raises(AssertionError, match="does not appear"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})


def test_hdf5_writer_get_object_dtype_errors(hdf5_output):
    """Object dtype discovery should reject unsupported attribute types."""

    class BadObject:
        def as_dict(self, lite=False):
            return {"bad": {"x": 1}}

    writer = HDF5Writer(hdf5_output, overwrite=True)
    with pytest.raises(ValueError, match="unrecognized type"):
        writer.get_object_dtype(BadObject())


def test_hdf5_writer_initializes_dummy_objects(hdf5_output):
    """Configured dummy datasets should be initialized as empty SPINE objects."""
    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"dummy": "RunInfo"})
    assert isinstance(writer.dummy_ds["dummy"], RunInfo)


def test_hdf5_writer_get_stored_keys_ready_and_dummy(hdf5_output):
    """Stored-key discovery should reuse ready keys and support dummy datasets."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer.ready = True
    writer.keys = {"index", "a"}
    assert writer.get_stored_keys({"index": [0], "a": [1]}) == {"index", "a"}

    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"dummy": "RunInfo"})
    keys = writer.get_stored_keys({"index": [0], "a": [1]})
    assert "dummy" in keys

    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"a": "RunInfo"})
    with pytest.raises(AssertionError, match="already exists"):
        writer.get_stored_keys({"index": [0], "a": [1]})


def test_hdf5_writer_get_stored_keys_rejects_unknown_skip_key(hdf5_output):
    """Stored-key discovery should reject skip keys that are not present in the data."""
    writer = HDF5Writer(hdf5_output, overwrite=True, skip_keys={"missing"})

    with pytest.raises(KeyError, match="skip_keys"):
        writer.get_stored_keys({"index": [0], "a": [1]})


def test_hdf5_writer_get_stored_keys_removes_requested_keys(hdf5_output):
    """Stored-key discovery should drop requested keys when they are present."""
    writer = HDF5Writer(hdf5_output, overwrite=True, skip_keys=["a"])
    keys = writer.get_stored_keys({"index": [0], "a": [1], "b": [2]})

    assert keys == {"index", "b"}


def test_hdf5_writer_get_data_type_special_cases(hdf5_output):
    """Data-type discovery should handle strings, ragged ndarrays, and unsupported containers."""
    writer = HDF5Writer(hdf5_output, overwrite=True)

    class BadContainer:
        dtype = object

        def __len__(self):
            return 1

    scalar_fmt = writer.get_data_type({"text": "hello"}, "text")
    assert scalar_fmt.scalar is True

    list_fmt = writer.get_data_type({"text": ["hello"]}, "text")
    assert list_fmt.scalar is True

    ragged_fmt = writer.get_data_type(
        {
            "jagged": [
                [np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)]
            ]
        },
        "jagged",
    )
    assert ragged_fmt.width == [2, 3]
    assert ragged_fmt.merge is False

    with pytest.raises(TypeError, match="Cannot store output"):
        writer.get_data_type({"bad": [BadContainer()]}, "bad")


def test_hdf5_writer_create_stores_cfg(hdf5_output):
    """Writer creation should persist the configuration when provided."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    data = {"index": np.array([0]), "value": [np.asarray([1.0], dtype=np.float32)]}
    writer.create(data, cfg={"io": {"loader": {}}})
    writer._ensure_file(0)
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "cfg" in out_file["info"].attrs


def test_hdf5_writer_creates_split_outputs_lazily(tmp_path):
    """Split outputs should only be created when a file receives data."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.finalize()
    writer.close()

    assert not os.path.exists(os.path.join(tmp_path, "split_0.h5"))
    assert os.path.exists(os.path.join(tmp_path, "split_1.h5"))


def test_hdf5_writer_reuses_open_handles(monkeypatch, hdf5_output):
    """Repeated writes with one writer instance should reuse the same append handle."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    first_open_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls == first_open_calls
    writer.close()


def test_hdf5_writer_keep_open_false_opens_per_write(monkeypatch, hdf5_output):
    """Disabling persistent handles should reopen the output file each write."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True, keep_open=False)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    first_write_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls > first_write_calls
    assert writer._file_handles == {}


def test_hdf5_writer_reopens_invalid_persistent_handle(monkeypatch, hdf5_output):
    """Invalid persistent handles should be reopened lazily on the next write."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer._file_handles[0].close()
    first_open_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls > first_open_calls
    writer.close()


def test_hdf5_writer_rejects_pid_reuse(monkeypatch, hdf5_output):
    """Persistent writer handles should not be reused across process boundaries."""
    pids = iter([100, 100, 200])
    monkeypatch.setattr("spine.io.write.hdf5.os.getpid", lambda: next(pids))

    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )

    with pytest.raises(RuntimeError, match="process-local"):
        writer(
            {
                "index": np.asarray([1]),
                "value": [np.asarray([2.0], dtype=np.float32)],
            }
        )


def test_hdf5_writer_context_manager_and_flush(hdf5_output):
    """Context-managed writers should flush and close persistent handles."""
    with HDF5Writer(hdf5_output, overwrite=True) as writer:
        writer(
            {
                "index": np.asarray([0]),
                "value": [np.asarray([1.0], dtype=np.float32)],
            }
        )
        assert writer._file_handles
        writer.flush()

    assert writer._file_handles == {}
    assert writer._handle_pid is None
    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_append_existing_file_keep_open_false_marks_incomplete(hdf5_output):
    """Appending to an existing file should reopen it and mark it incomplete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.finalize()
    writer.close()

    writer = HDF5Writer(hdf5_output, append=True, keep_open=False)
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_flush_frequency_flushes_persistent_handle(
    monkeypatch, hdf5_output
):
    """Flush frequency should flush persistent handles after the requested interval."""
    writer = HDF5Writer(hdf5_output, overwrite=True, flush_frequency=1)
    flush_calls = 0

    original_flush = h5py.File.flush

    def counted_flush(handle):
        nonlocal flush_calls
        flush_calls += 1
        return original_flush(handle)

    monkeypatch.setattr(h5py.File, "flush", counted_flush)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert flush_calls >= 1


def test_hdf5_writer_close_swallows_handle_close_errors(hdf5_output):
    """Writer cleanup should clear state even if a handle raises on close."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer._handle_pid = 123

    class BadHandle:
        def close(self):
            raise OSError("boom")

    writer._file_handles[0] = BadHandle()
    writer.close()

    assert writer._file_handles == {}
    assert writer._handle_pid is None


def test_hdf5_writer_call_scalar_split_and_dummy(tmp_path):
    """Writer call should wrap scalar inputs before hitting dummy-dataset validation."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name,
        prefix=["a", "b"],
        split=True,
        overwrite=True,
        dummy_ds={"dummy": "RunInfo"},
    )
    with pytest.raises(AssertionError, match="dummy dataset dummy already exists"):
        writer(
            {"index": 0, "file_index": 1, "value": np.asarray([1.0], dtype=np.float32)}
        )


def test_hdf5_writer_call_splits_entries_by_file_index(tmp_path):
    """Writer calls should fan entries out across split outputs using file_index."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "file_index": np.asarray([0, 1], dtype=np.int64),
            "value": [
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1


def test_hdf5_writer_call_splits_entries_by_file_index_without_persistent_handles(
    tmp_path,
):
    """Split writes should also close transient per-file handles when keep_open is disabled."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name,
        prefix=["a", "b"],
        split=True,
        overwrite=True,
        keep_open=False,
    )
    writer(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "file_index": np.asarray([0, 1], dtype=np.int64),
            "value": [
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ],
        }
    )
    writer.finalize()

    assert writer._file_handles == {}
    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1


def test_hdf5_writer_auto_finalizes_split_predecessors(tmp_path):
    """Sequential split writing should finalize earlier files once the writer advances."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert out_file["info"].attrs["complete"]
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_auto_finalizes_split_predecessors_without_persistent_handles(
    tmp_path,
):
    """Sequential split writing should finalize predecessors without persistent handles."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name, prefix=["a", "b"], split=True, overwrite=True, keep_open=False
    )
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_disables_sequential_split_finalization_on_out_of_order_writes(
    tmp_path,
):
    """Out-of-order split writes should disable automatic predecessor finalization."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert not writer._split_sequential


def test_hdf5_writer_keeps_split_predecessor_guard_on_same_file_id(tmp_path):
    """Repeated writes to the same split file should not finalize predecessors."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert writer._split_sequential
    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_rejects_writes_to_finalized_file(tmp_path):
    """Finalized files should not accept more writes."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.finalize()

    with pytest.raises(RuntimeError, match="already finalized"):
        writer._ensure_file(0)


def test_hdf5_writer_store_jagged_and_scalar_append_key(hdf5_output):
    """Append-key should cover scalar fanout and jagged list storage."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    data = {
        "index": np.array([0]),
        "scalar": 5,
        "jagged": [
            [np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)]
        ],
    }
    writer.create(data)
    writer._ensure_file(0)

    with h5py.File(hdf5_output, "a") as out_file:
        event = np.empty(1, writer.event_dtype)
        writer.append_key(out_file, event, data, "scalar", 0)
        writer.append_key(out_file, event, data, "jagged", 0)

        assert out_file["scalar"].shape[0] == 1
        assert isinstance(out_file["jagged"], h5py.Group)
        assert out_file["jagged"]["index"].shape[0] == 1


def generate_object_list(cls, sizes):
    """Generates a dummy list of lists of objects of the request class.

    Parameters
    ----------
    cls : object
        Class that the objects should belong to
    sizes : List[int]
        Number of objects in each list

    Returns
    -------
    List[ObjectList[obj]]
        List of typed lists of objects
    """
    return [ObjectList([cls() for _ in range(s)], cls()) for s in sizes]
