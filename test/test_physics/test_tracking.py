import numpy as np
import pytest

from spine.data import TensorData
from spine.physics.tracking import (
    check_track_orientation,
    check_track_orientation_ppn,
    get_track_deposition_gradient,
    get_track_length,
    get_track_segment_dedxs,
    get_track_segments,
    get_track_spline,
)
from spine.utils.ppn import ppn_prediction_schema


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

    sparse_segments = get_track_segments(
        points, segment_length=2.0, method="bin_pca", min_count=0
    )
    assert any(len(segment) == 0 for segment in sparse_segments[0])


def line_track(count=11):
    """Return a straight track with a rising deposition profile."""
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 0] = np.arange(count)
    values = np.arange(1, count + 1, dtype=np.float32)
    return points, values


def test_ppn_track_orientation_empty_shared_and_distinct_candidates():
    """Endpoint scores should handle empty, shared, and distinct PPN matches."""
    empty = TensorData(
        coords=np.empty((0, 3), dtype=np.float32),
        features=np.empty((0, 11), dtype=np.float32),
        schema=ppn_prediction_schema(True),
    )
    assert check_track_orientation_ppn(np.zeros(3), np.ones(3), empty)

    features = np.zeros((2, 11), dtype=np.float32)
    features[:, 9:11] = [[0.9, 0.1], [0.1, 0.9]]
    candidates = TensorData(
        coords=np.array([[0.1, 0, 0], [9.9, 0, 0]], dtype=np.float32),
        features=features,
        schema=ppn_prediction_schema(True),
    )
    assert check_track_orientation_ppn(np.zeros(3), np.array([10, 0, 0]), candidates)

    shared = TensorData(
        coords=np.array([[1, 0, 0]], dtype=np.float32),
        features=features[:1],
        schema=ppn_prediction_schema(True),
    )
    assert check_track_orientation_ppn(np.zeros(3), np.array([10, 0, 0]), shared)
    shared.features[0, 9:11] = [0.1, 0.9]
    assert not check_track_orientation_ppn(np.zeros(3), np.array([10, 0, 0]), shared)


def test_track_length_orientation_and_gradients():
    """Track length and orientation estimators should cover every mode."""
    points, values = line_track()
    start, end = points[0], points[-1]
    assert get_track_length(points, method="displacement") == pytest.approx(10.0)
    for method in ("step", "step_next", "bin_pca"):
        assert (
            get_track_length(
                points, segment_length=3.0, point=start, method=method, min_count=1
            )
            > 0.0
        )
    assert get_track_length(points, 3.0, method="splines") > 0.0
    with pytest.raises(ValueError, match="not recognized"):
        get_track_length(points, method="bad")

    assert isinstance(
        bool(check_track_orientation(points, values, start, end, method="local")), bool
    )
    assert isinstance(
        bool(
            check_track_orientation(
                points, values, start, end, method="local", anchor_points=False
            )
        ),
        bool,
    )
    assert isinstance(
        bool(
            check_track_orientation(
                points,
                values,
                start,
                end,
                method="gradient",
                segment_length=3.0,
                segment_min_count=1,
            )
        ),
        bool,
    )
    with pytest.raises(ValueError, match="not recognized"):
        check_track_orientation(points, values, start, end, method="bad")

    gradient, dedxs, ranges, lengths = get_track_deposition_gradient(
        points, values, start, segment_length=3.0, min_count=1
    )
    assert np.isfinite(gradient)
    assert len(dedxs) == len(ranges)
    assert len(lengths) >= len(dedxs)
    empty_gradient = get_track_deposition_gradient(
        points, values, start, segment_length=3.0, min_count=100
    )
    assert empty_gradient[0] == 0.0


def test_track_segmentation_dedx_and_splines():
    """Segment builders should cover point anchoring, PCA bins, and spline fallbacks."""
    points, values = line_track()
    for method, point, anchor in [
        ("step", points[0] + 0.2, True),
        ("step_next", points[0], False),
        ("step_next", None, True),
        ("bin_pca", points[-1], True),
    ]:
        segments, directions, lengths = get_track_segments(
            points, 3.0, point, method, anchor, min_count=1
        )
        assert len(segments) == len(directions) == len(lengths)
    with pytest.raises(ValueError, match="not recognized"):
        get_track_segments(points, 3.0, method="bad")

    dedxs, errors, ranges, segments, directions, lengths = get_track_segment_dedxs(
        points,
        values,
        points[0],
        segment_length=3.0,
        method="step_next",
        min_count=1,
    )
    assert len(dedxs) == len(errors) == len(ranges) == len(segments)
    assert directions.shape[0] == lengths.shape[0] == len(segments)
    invalid = get_track_segment_dedxs(
        points, values, points[0], segment_length=3.0, min_count=100
    )
    assert np.all(invalid[0] == -1.0)

    short = get_track_spline(points[:3], 2.0)
    assert isinstance(short[0], np.ndarray)
    assert short[-1] == pytest.approx(2.0)
    assert get_track_spline(points, 100.0)[-1] == pytest.approx(10.0)
    assert get_track_spline(points, 2.0, s=0.0)[-1] == pytest.approx(10.0)


def test_step_segments_handle_zero_direction_and_backward_remainder():
    """Stepping should terminate safely for coincident and backward-only points."""
    coincident = np.zeros((2, 3), dtype=np.float32)
    segments = get_track_segments(
        coincident, 1.0, coincident[0], method="step", min_count=0
    )
    np.testing.assert_array_equal(segments[1][0], [1.0, 0.0, 0.0])

    points = np.array([[0, 0, 0], [1, 0, 0], [-3, 0, 0]], dtype=np.float32)
    segments = get_track_segments(
        points, 1.5, points[0], method="step", anchor_point=False, min_count=0
    )
    assert len(segments[0]) == 1
