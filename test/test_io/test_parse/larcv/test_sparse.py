"""Test that the sparse data parsers work as intended."""

import numpy as np
import pytest

from spine.data.larcv import Meta
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.parse.data import ParserTensor
from spine.io.parse.larcv.sparse import *
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

    call_parser = Sparse2DParser(
        dtype="float32", sparse_event="sparse", projection_id=projection_id
    )
    call_result = call_parser({"sparse": sparse2d_event})
    assert isinstance(call_result, ParserTensor)


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


@pytest.mark.parametrize("sparse3d_event_list", [6], indirect=True)
def test_sparse_parser_call_paths(
    sparse3d_event, sparse3d_event_list, sparse3d_seg_event
):
    """Wrapper calls should route named inputs through the sparse parsers."""
    sparse_parser = Sparse3DParser(dtype="float32", sparse_event="sparse")
    assert isinstance(sparse_parser({"sparse": sparse3d_event}), ParserTensor)

    aggr_parser = Sparse3DAggregateParser(
        dtype="float32", sparse_event_list=["s0", "s1"], aggr="sum"
    )
    assert isinstance(
        aggr_parser({"s0": sparse3d_event_list[0], "s1": sparse3d_event_list[1]}),
        ParserTensor,
    )

    rescale_parser = Sparse3DChargeRescaledParser(
        dtype="float32",
        sparse_event_list=[f"s{i}" for i in range(7)],
    )
    assert isinstance(
        rescale_parser(
            {
                f"s{i}": value
                for i, value in enumerate(
                    sparse3d_event_list[:6] + [sparse3d_seg_event]
                )
            }
        ),
        ParserTensor,
    )

    ghost_parser = Sparse3DGhostParser(dtype="float32", sparse_event="seg")
    assert isinstance(ghost_parser({"seg": sparse3d_seg_event}), ParserTensor)


@pytest.mark.parametrize("sparse3d_event_list", [2], indirect=True)
def test_sparse3d_parser_constructor_validation_and_options(
    sparse3d_event, sparse3d_event_list
):
    """Sparse parser construction should validate feature configuration eagerly."""
    with pytest.raises(ValueError, match="No need to lexsort"):
        Sparse3DParser(dtype="float32", sparse_event=sparse3d_event, lexsort=True)

    with pytest.raises(ValueError, match="nhits_idx"):
        Sparse3DParser(
            dtype="float32",
            sparse_event_list=sparse3d_event_list[:1],
            hit_keys=[0],
        )

    with pytest.raises(ValueError, match="divider"):
        Sparse3DParser(
            dtype="float32",
            sparse_event_list=sparse3d_event_list[:2],
            num_features=3,
        )

    parser = Sparse3DParser(
        dtype="float32",
        sparse_event_list=sparse3d_event_list[:2],
        num_features=1,
        index_cols=[0],
        sum_cols=[0],
        lexsort=True,
    )
    result = parser.process(sparse_event_list=sparse3d_event_list[:2])

    assert isinstance(result, ParserTensor)
    assert np.array_equal(parser.index_cols, np.asarray([0]))
    assert np.array_equal(parser.sum_cols, np.asarray([0]))


@pytest.mark.parametrize("sparse3d_event_list", [2], indirect=True)
def test_sparse3d_parser_computes_nhits_and_validates_index(sparse3d_event_list):
    """Sparse3DParser should add nhits features and reject invalid insertion indices."""
    parser = Sparse3DParser(
        dtype="float32",
        sparse_event_list=sparse3d_event_list,
        num_features=2,
        hit_keys=[0, 1],
        nhits_idx=1,
    )
    result = parser.process(sparse_event_list=sparse3d_event_list)
    assert isinstance(result, ParserTensor)
    assert result.features.shape[1] == 3

    parser = Sparse3DParser(
        dtype="float32",
        sparse_event_list=sparse3d_event_list,
        num_features=2,
        hit_keys=[0, 1],
        nhits_idx=3,
    )
    with pytest.raises(ValueError, match="out of range"):
        parser.process(sparse_event_list=sparse3d_event_list)
