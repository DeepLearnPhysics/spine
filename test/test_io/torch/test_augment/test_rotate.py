"""Tests for the rotate augmenter."""

from .helpers import GeoManager, RotateAugment, make_meta, make_tensor, np, pytest


def test_rotate_augment_default_uses_image_frame_rotation():
    """Default rotation should preserve the historical voxel-frame behavior."""
    meta = make_meta()
    tensor = make_tensor([[0, 0, 0], [1, 3, 1]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = RotateAugment(axes=(0, 1), k=1)
    result, rot_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 0, 0], [0, 1, 1]]))
    assert np.array_equal(rot_meta.count, np.asarray([4, 2, 2]))
    assert np.array_equal(rot_meta.lower, meta.lower)


def test_rotate_augment_explicit_center_rotates_about_pivot():
    """Explicit centers should support moving-meta behavior explicitly."""
    meta = make_meta()
    tensor = make_tensor([[0, 0, 0], [1, 3, 1]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = RotateAugment(
        axes=(0, 1),
        k=1,
        center=np.asarray([1.0, 2.0, 1.0]),
        keep_meta=False,
    )
    result, rot_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 0, 0], [0, 1, 1]]))
    assert np.allclose(rot_meta.lower, np.asarray([-1.0, 1.0, 0.0]))
    assert np.allclose(rot_meta.upper, np.asarray([3.0, 3.0, 2.0]))


def test_rotate_augment_explicit_center_can_keep_meta_fixed():
    """Centered rotation should optionally keep the detector frame fixed."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 2.0))
    tensor = make_tensor([[0, 0, 0], [3, 3, 0]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = RotateAugment(
        axes=(0, 1),
        k=1,
        center=np.asarray([1.0, 2.0, 1.0]),
        keep_meta=True,
    )
    result, rot_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[2, 1, 0]]))
    assert np.array_equal(
        result["voxels"].features, np.asarray([[0.0]], dtype=np.float32)
    )
    assert rot_meta is meta


def test_rotate_augment_can_use_detector_center():
    """Geometry-based rotation centers should use the detector TPC center."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    pivot = geo.tpc.center
    meta = make_meta(
        lower=pivot + np.asarray([-1.0, -2.0, -1.0]),
        upper=pivot + np.asarray([1.0, 2.0, 1.0]),
    )
    tensor = make_tensor([[0, 0, 0], [1, 3, 1]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = RotateAugment(axes=(0, 1), k=1, use_geo_center=True, keep_meta=False)
    result, rot_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 0, 0], [0, 1, 1]]))
    assert np.allclose((rot_meta.lower + rot_meta.upper) / 2.0, pivot)
    assert np.allclose(rot_meta.lower, pivot + np.asarray([-2.0, -1.0, -1.0]))
    assert np.allclose(rot_meta.upper, pivot + np.asarray([2.0, 1.0, 1.0]))

    GeoManager.reset()


def test_rotate_constructor_validates_arguments():
    with pytest.raises(ValueError):
        RotateAugment(axes=(0,))
    with pytest.raises(ValueError):
        RotateAugment(axes=(1, 1))
    with pytest.raises(ValueError):
        RotateAugment(axes=(0, 3))
    with pytest.raises(ValueError):
        RotateAugment(k=1.5)


def test_rotate_with_zero_turns_returns_inputs_unchanged():
    meta = make_meta()
    tensor = make_tensor([[0, 0, 0]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = RotateAugment(k=0)
    result, rot_meta = augment(data, meta, ["voxels", "meta"], {})

    assert result is data
    assert rot_meta is meta
    assert result["meta"] is meta


def test_rotate_sample_k_uses_random_draw_when_unset():
    augment = RotateAugment()
    np.random.seed(3)
    value = augment.sample_k()
    assert value in (0, 1, 2, 3)


def test_rotate_coords_supports_half_and_three_quarter_turns():
    augment = RotateAugment(axes=(0, 1), k=1)
    coords = np.asarray([[0, 0, 0], [1, 3, 1]], dtype=np.int64)
    count = np.asarray([2, 4, 2], dtype=np.int64)

    half = augment.rotate_coords(coords, count, 2)
    three_quarter = augment.rotate_coords(coords, count, 3)

    assert np.array_equal(half, np.asarray([[1, 3, 0], [0, 0, 1]]))
    assert np.array_equal(three_quarter, np.asarray([[0, 1, 0], [3, 0, 1]]))


def test_rotate_points_supports_half_and_three_quarter_turns():
    augment = RotateAugment(axes=(0, 1), k=1)
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 3.0, 1.0]], dtype=np.float32)
    pivot = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)

    half = augment.rotate_points(points, pivot, 2)
    three_quarter = augment.rotate_points(points, pivot, 3)

    assert np.allclose(
        half, np.asarray([[2.0, 2.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float32)
    )
    assert np.allclose(
        three_quarter,
        np.asarray([[0.0, 2.0, 0.0], [3.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_rotate_generate_meta_handles_half_turn_without_axis_swap():
    meta = make_meta(lower=(1.0, 2.0, 3.0), upper=(3.0, 6.0, 5.0))
    augment = RotateAugment(axes=(0, 1), k=2)
    rot_meta = augment.generate_meta(meta, 2)

    assert np.array_equal(rot_meta.count, meta.count)
    assert np.array_equal(rot_meta.size, meta.size)
    assert np.array_equal(rot_meta.lower, meta.lower)
