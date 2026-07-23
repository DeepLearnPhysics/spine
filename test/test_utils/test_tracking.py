import numpy as np
import pytest

from spine.utils.tracking import get_track_length, get_track_segments


def test_bin_pca_segments_fall_back_for_one_point_chunks():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    segments, dirs, lengths = get_track_segments(
        points,
        segment_length=5.0,
        method="bin_pca",
        min_count=0,
    )

    assert len(segments) == 2
    assert dirs.shape == (2, 3)
    assert lengths.shape == (2,)


@pytest.mark.parametrize("anchor_point", [False, True])
def test_step_segments_accept_endpoint_with_different_precision(anchor_point):
    """Endpoint precision should be cast to the coordinate precision."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    point = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    segments, dirs, lengths = get_track_segments(
        points,
        segment_length=1.5,
        point=point,
        method="step_next",
        anchor_point=anchor_point,
        min_count=0,
    )

    assert len(segments)
    assert dirs.dtype == points.dtype
    assert lengths.dtype == points.dtype

    length = get_track_length(
        points,
        segment_length=1.5,
        point=point,
        method="step_next",
        anchor_point=anchor_point,
        min_count=0,
    )
    assert np.isfinite(length)
