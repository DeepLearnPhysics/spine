import numpy as np

from spine.utils.tracking import get_track_segments, get_track_spline


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


def test_track_spline_handles_repeated_longitudinal_coordinates():
    """Repeated spline abscissas should be combined before fitting."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    u, spline_points, _, length = get_track_spline(
        points,
        segment_length=1.0,
        s=0.0,
    )

    assert len(u) == len(points)
    assert np.all(np.isfinite(spline_points))
    assert np.isclose(length, 3.0)
