"""Event reconstruction coverage for batched index products."""

import numpy as np

from spine.data import (
    EdgeIndexBatch,
    EdgeIndexData,
    IndexBatch,
    IndexData,
    IndexListData,
)


def test_flat_and_jagged_index_events_preserve_spans():
    """Index event extraction should remove offsets and retain parent spans."""
    flat = IndexBatch(np.asarray([0, 2, 5, 7]), spans=[5, 4], counts=[2, 2])
    flat_event = flat.event(1)

    assert isinstance(flat_event, IndexData)
    assert flat_event.span == 4
    np.testing.assert_array_equal(flat_event.features, [0, 2])

    listed = IndexBatch(
        [np.asarray([0, 2]), np.asarray([5]), np.asarray([6, 7])],
        spans=[5, 4],
        counts=[1, 2],
        single_counts=[2, 1, 2],
    )
    list_event = listed.event(1)

    assert isinstance(list_event, IndexListData)
    assert list_event.span == 4
    np.testing.assert_array_equal(list_event.single_counts, [1, 2])
    np.testing.assert_array_equal(list_event.features[0], [0])
    np.testing.assert_array_equal(list_event.features[1], [1, 2])


def test_edge_index_event_preserves_orientation_and_span():
    """Edge event extraction should restore canonical local incidence form."""
    batch = EdgeIndexBatch(
        np.asarray([[0, 1, 3, 4], [1, 0, 4, 3]]),
        counts=[2, 2],
        spans=[3, 2],
        directed=False,
    )
    event = batch.event(1)

    assert isinstance(event, EdgeIndexData)
    assert event.span == 2
    assert event.directed is False
    np.testing.assert_array_equal(event.index, [[0, 1], [1, 0]])
