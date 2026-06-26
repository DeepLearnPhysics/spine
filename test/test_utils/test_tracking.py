import numpy as np

from spine.utils.tracking import get_track_segments


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
