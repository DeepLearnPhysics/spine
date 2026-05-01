"""Tests for the LArCV reader."""

import pytest

from spine.io.read import LArCVReader
from spine.utils.conditional import ROOT, ROOT_AVAILABLE


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT is required to read LArCV files.")
def test_larcv_reader(larcv_data):
    """Tests the loading of a LArCV file."""
    # Get the list of tree keys in the larcv file
    root_file = ROOT.TFile(larcv_data, "r")
    num_entries = None
    tree_keys = []
    for tree in root_file.GetListOfKeys():
        tree_keys.append(tree.GetName().split("_tree")[0])
        if num_entries is None:
            num_entries = root_file.Get(tree.GetName()).GetEntries()

    root_file.Close()

    # Intialize the reader
    reader = LArCVReader(
        file_keys=larcv_data,
        tree_keys=tree_keys,
        create_run_map=True,
        run_info_key=tree_keys[0],
    )

    # Check that the number of events in the dataset is as expected
    assert reader.num_entries == num_entries

    # Load every entry, check that they contain what is expected
    for entry in reader:
        for key in tree_keys:
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
    reader = LArCVReader(file_keys=[larcv_data, larcv_data], tree_keys=tree_keys[:1])
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
    reader = LArCVReader(
        file_keys=[larcv_data, larcv_data], tree_keys=tree_keys[:1], limit_num_files=1
    )
    assert reader.num_entries == num_entries
