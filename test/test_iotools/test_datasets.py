"""Test that the dataset classes work as intended."""

import pytest

import ROOT

from mlreco.iotools.datasets import *


def test_larcv_dataset(larcv_data):
    """Tests a torch dataset based on LArCV data.

    Most of the functions of this dataset are shared with the underlying
    :class:`LArCVReader` class which is tested elsewhere.
    """
    # Get the list of tree keys in the larcv file
    root_file = ROOT.TFile(larcv_data, 'r')
    num_entries = None
    tree_keys = []
    for tree in root_file.GetListOfKeys():
        tree_keys.append(tree.GetName().split('_tree')[0])
        if num_entries is None:
            num_entries = getattr(root_file, tree.GetName()).GetEntries()

    root_file.Close()

    # Create a dummy schema based on the data keys
    schema = {}
    for key in tree_keys:
        datatype = key.split('_')[0]
        el = {}
        if datatype == 'sparse3d':
            el['parser'] = f'parse_sparse3d'
            el['sparse_event'] = key
        elif datatype == 'cluster3d':
            el['parser'] = f'parse_cluster3d'
            el['cluster_event'] = key
        elif datatype == 'particle':
            el['parser'] = f'parse_particles'
            el['particle_event'] = key
            el['pixel_coordinates'] = False
        else:
            raise ValueError("Unrecognized data product, cannot set up schema.")

        schema[key] = el

    # Initialize the dataset
    dataset = LArCVDataset(file_keys=larcv_data, schema=schema)

    # Load the items in the dataset, check the keys
    for i in range(len(dataset)):
        entry = dataset[i]
        for key in tree_keys:
            assert key in entry
        assert 'index' in entry
        assert entry['index'] == i

    # Check that the data keys are as expected
    for key in tree_keys:
        assert key in dataset.data_keys()

    # Check that one can list the content of the dataset
    data_dict = dataset.list_data(larcv_data)
    data_keys = []
    for val in data_dict.values():
        data_keys += list(val)
    for key in tree_keys:
        assert key in data_keys
