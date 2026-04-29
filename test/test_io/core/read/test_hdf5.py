"""Tests for the HDF5 reader."""

import h5py
import numpy as np

from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.core.read import HDF5Reader


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
