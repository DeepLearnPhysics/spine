"""Test that the reader classes work as intended."""

import pytest

import numpy as np
import ROOT
import h5py

from spine.io.read import *


def test_larcv_reader(larcv_data):
    """Tests the loading of a LArCV file."""
    # Get the list of tree keys in the larcv file
    root_file = ROOT.TFile(larcv_data, "r")
    num_entries = None
    tree_keys = []
    for tree in root_file.GetListOfKeys():
        tree_keys.append(tree.GetName().split("_tree")[0])
        if num_entries is None:
            num_entries = getattr(root_file, tree.GetName()).GetEntries()

    root_file.Close()

    # Intialize the reader
    reader = LArCVReader(
        larcv_data, tree_keys, create_run_map=True, run_info_key=tree_keys[0]
    )

    # Check that the number of events in the dataset is as expected
    assert reader.num_entries == num_entries

    # Load every entry, check that they contain what is expected
    for i in range(len(reader)):
        entry = reader[i]
        for key in tree_keys:
            assert key in entry

    # Check that the run map exists
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
    reader = LArCVReader([larcv_data, larcv_data], tree_keys[:1])
    assert reader.num_entries == 2 * num_entries
    for i in range(len(reader)):
        reader[i]

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
    reader = LArCVReader([larcv_data, larcv_data], tree_keys[:1], limit_num_files=1)
    assert reader.num_entries == num_entries


def test_hdf5_reader(hdf5_data):
    """Tests the loading of a LArCV file."""
    # Get the list of tree keys in the HDF5 file
    data_keys = None
    with h5py.File(hdf5_data, "r") as h5_file:
        data_keys = list(h5_file.keys())
        num_entries = len(h5_file["events"])

    # Intialize the reader
    reader = HDF5Reader(hdf5_data, create_run_map=True)

    # Check that the number of events in the dataset is as expected
    assert reader.num_entries == num_entries

    # Load every entry, check that they contain what is expected
    for i in range(len(reader)):
        entry = reader[i]
        for key in data_keys:
            if key not in ["info", "events"]:
                assert key in entry

    # Check that the run map exists
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
    reader = HDF5Reader([hdf5_data, hdf5_data])
    assert reader.num_entries == 2 * num_entries
    for i in range(len(reader)):
        reader[i]

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
