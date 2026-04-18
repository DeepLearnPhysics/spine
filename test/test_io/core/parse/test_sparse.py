"""Test that the sparse data parsers work as intended."""

import pytest

from spine.data.larcv import Meta
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.core.parse.data import ParserTensor
from spine.io.core.parse.sparse import *
from spine.utils.conditional import LARCV_AVAILABLE

pytestmark = pytest.mark.skipif(
    not LARCV_AVAILABLE, reason="LArCV is required to generate parser fixtures."
)


@pytest.mark.parametrize("projection_id", [0, 1, 2])
def test_parse_sparse2d(sparse2d_event, projection_id):
    """Tests the parsing of LArCV 2D sparse data."""
    # Initialize the parser
    parser = Sparse2DParser(
        dtype="float32", sparse_event=sparse2d_event, projection_id=projection_id
    )

    # Parse the data
    result = parser.process(sparse_event=sparse2d_event)

    # There should be 3 components of the output
    # - The first has both coordinates for each point
    # - The second has the feature tensor
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 2
    assert result.features.shape[1] == 1
    assert isinstance(result.meta, ImageMeta2D)


@pytest.mark.parametrize("projection_id", [0, 1, 2])
@pytest.mark.parametrize("sparse2d_event_list", [1, 2], indirect=True)
def test_parse_sparse2d_list(sparse2d_event_list, projection_id):
    """Tests the parsing of a LArCV 2D sparse data list (multi-features)."""
    # Initialize the parser
    parser = Sparse2DParser(
        dtype="float32",
        sparse_event_list=sparse2d_event_list,
        projection_id=projection_id,
    )

    # Parse the data
    result = parser.process(sparse_event_list=sparse2d_event_list)

    # There should be 3 components of the output
    # - The first has all 2 coordinates for each point
    # - The second has the feature tensor (one per input tensor)
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 2
    assert result.features.shape[1] == len(sparse2d_event_list)
    assert isinstance(result.meta, ImageMeta2D)


def test_parse_sparse3d(sparse3d_event):
    """Tests the parsing of LArCV 3D sparse data."""
    # Initialize the parser
    parser = Sparse3DParser(dtype="float32", sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == 1
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("num_features", [None, 1])
@pytest.mark.parametrize("sparse3d_event_list", [1, 2], indirect=True)
def test_parse_sparse3d_list(sparse3d_event_list, num_features):
    """Tests the parsing of a LArCV 3D sparse data list (multi-features)."""
    # Initialize the parser
    parser = Sparse3DParser(
        dtype="float32",
        sparse_event_list=sparse3d_event_list,
        num_features=num_features,
    )

    # Parse the data
    result = parser.process(sparse_event_list=sparse3d_event_list)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (one per input tensor)
    # - The third has the metadata
    div = len(sparse3d_event_list) / num_features if num_features else 1
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == len(sparse3d_event_list) / div
    assert isinstance(result.meta, ImageMeta3D)


def test_parse_sparse3d_ghost(sparse3d_seg_event):
    """Tests the parsing of LArCV 3D sparse semantic labels to ghost labels."""
    # Initialize the parser
    parser = Sparse3DGhostParser(dtype="float32", sparse_event=sparse3d_seg_event)

    # Parse the data
    result = parser.process_ghost(sparse_event=sparse3d_seg_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the ghost labels
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == 1
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("collection_only", [False, True])
@pytest.mark.parametrize("sparse3d_event_list", [6], indirect=True)
def test_parse_spars3d_rescale(
    sparse3d_event_list, sparse3d_seg_event, collection_only
):
    """Tests the parsing of 3D LArCV sparse data into a set of rescaled charges.

    This parser takes 6 values (3 charges, 3 indexes) and combines this
    with segementation labels to produce a single rescaled charge feature.
    """
    # Merge the list of random values with the segmentation
    sparse3d_event_list += [sparse3d_seg_event]

    # Initialize the parser
    parser = Sparse3DChargeRescaledParser(
        dtype="float32",
        sparse_event_list=sparse3d_event_list,
        collection_only=collection_only,
    )

    # Parse the data
    result = parser.process_rescale(sparse_event_list=sparse3d_event_list)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the rescaled charge
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == 1
    assert isinstance(result.meta, ImageMeta3D)
