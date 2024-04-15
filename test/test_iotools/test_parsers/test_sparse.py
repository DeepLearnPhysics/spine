"""Test that the sparse data parsers work as intended."""

import pytest

from mlreco import Meta
from mlreco.iotools.parsers.sparse import *

from test.test_iotools.test_parsers.fixtures import (
        fixture_sparse2d_event, fixture_sparse2d_event_list,
        fixture_sparse3d_event, fixture_sparse3d_event_list,
        fixture_sparse3d_seg_event)


@pytest.mark.parametrize('projection_id', [0, 1, 2])
def test_parse_sparse2d(sparse2d_event, projection_id):
    """Tests the parsing of LArCV 2D sparse data."""
    # Initialize the parser
    parser = Sparse2DParser(
            sparse_event=sparse2d_event, projection_id=projection_id)

    # Parse the data
    result = parser.process(sparse_event=sparse2d_event)

    # There should be 3 components of the output
    # - The first has both coordinates for each point
    # - The second has the feature tensor
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 2
    assert result[1].shape[1] == 1
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize('projection_id', [0, 1, 2])
@pytest.mark.parametrize('sparse2d_event_list', [1, 2], indirect=True)
def test_parse_sparse2d_list(sparse2d_event_list, projection_id):
    """Tests the parsing of a LArCV 2D sparse data list (multi-features)."""
    # Initialize the parser
    parser = Sparse2DParser(
            sparse_event_list=sparse2d_event_list, projection_id=projection_id)

    # Parse the data
    result = parser.process(sparse_event_list=sparse2d_event_list)

    # There should be 3 components of the output
    # - The first has all 2 coordinates for each point
    # - The second has the feature tensor (one per input tensor)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 2
    assert result[1].shape[1] == len(sparse2d_event_list)
    assert isinstance(result[2], Meta)


def test_parse_sparse3d(sparse3d_event):
    """Tests the parsing of LArCV 3D sparse data."""
    # Initialize the parser
    parser = Sparse3DParser(sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 1
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize('num_features', [None, 1])
@pytest.mark.parametrize('sparse3d_event_list', [1, 2], indirect=True)
def test_parse_sparse3d_list(sparse3d_event_list, num_features):
    """Tests the parsing of a LArCV 3D sparse data list (multi-features)."""
    # Initialize the parser
    parser = Sparse3DParser(
            sparse_event_list=sparse3d_event_list, num_features=num_features)

    # Parse the data
    result = parser.process(sparse_event_list=sparse3d_event_list)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (one per input tensor)
    # - The third has the metadata
    div = len(sparse3d_event_list)/num_features if num_features else 1
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == len(sparse3d_event_list)/div
    assert isinstance(result[2], Meta)


def test_parse_sparse3d_ghost(sparse3d_seg_event):
    """Tests the parsing of LArCV 3D sparse semantic labels to ghost labels."""
    # Initialize the parser
    parser = Sparse3DGhostParser(sparse_event=sparse3d_seg_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_seg_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the ghost labels
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 1
    assert ((result[1] == 0) | (result[1] == 1)).all()
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize('collection_only', [False, True])
@pytest.mark.parametrize('sparse3d_event_list', [6], indirect=True)
def test_parse_spars3d_rescale(sparse3d_event_list, sparse3d_seg_event,
                               collection_only):
    """Tests the parsing of 3D LArCV sparse data into a set of rescaled charges.
    
    This parser takes 6 values (3 charges, 3 indexes) and combines this
    with segementation labels to produce a single rescaled charge feature.
    """
    # Merge the list of random values with the segmentation
    sparse3d_event_list += [sparse3d_seg_event]

    # Initialize the parser
    parser = Sparse3DChargeRescaledParser(
            sparse_event_list=sparse3d_event_list,
            collection_only=collection_only)

    # Parse the data
    result = parser.process(sparse_event_list=sparse3d_event_list)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the rescaled charge
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 1
    assert isinstance(result[2], Meta)
