"""Tests for the flip augmenter."""

from .helpers import FlipAugment, GeoManager, make_meta, make_tensor, np, pytest


def test_flip_augment_reflects_about_requested_plane():
    """Flip should implement a true reflection, not another rotation."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = FlipAugment(axis=0)
    result, flip_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 1, 2], [0, 1, 2]]))
    assert np.allclose(flip_meta.lower, meta.lower)
    assert np.allclose(flip_meta.upper, meta.upper)


def test_flip_augment_can_use_detector_center():
    """Geometry-based centers should come from the detector TPC center."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    pivot = geo.tpc.center
    meta = make_meta(lower=pivot - 2.0, upper=pivot + 2.0)
    tensor = make_tensor([[0, 1, 1], [3, 1, 1]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = FlipAugment(axis=0, use_geo_center=True, keep_meta=False)
    result, flip_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 1, 1], [0, 1, 1]]))
    assert np.allclose((flip_meta.lower + flip_meta.upper) / 2.0, pivot)

    GeoManager.reset()


def test_flip_augment_explicit_center_can_keep_meta_fixed():
    """Explicit flip centers should optionally keep the detector frame fixed."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = FlipAugment(axis=0, center=np.asarray([1.0, 2.0, 2.0]), keep_meta=True)
    result, flip_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[1, 1, 2]]))
    assert np.array_equal(
        result["voxels"].features, np.asarray([[0.0]], dtype=np.float32)
    )
    assert flip_meta is meta


def test_flip_augment_explicit_center_preserves_moving_meta_by_default():
    """Explicit flip centers should support moving-meta behavior explicitly."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = FlipAugment(axis=0, center=np.asarray([1.0, 2.0, 2.0]), keep_meta=False)
    result, flip_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 1, 2], [0, 1, 2]]))
    assert np.allclose(flip_meta.lower, np.asarray([-2.0, 0.0, 0.0]))
    assert np.allclose(flip_meta.upper, np.asarray([2.0, 4.0, 4.0]))


def test_flip_augment_probability_zero_leaves_event_unchanged():
    """Flip probability can disable the transform for an event."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = FlipAugment(axis=0, p=0.0)
    result, flip_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(result["voxels"].coords, np.asarray([[0, 1, 2], [3, 1, 2]]))
    assert flip_meta is meta


def test_flip_augment_probability_uses_random_draw(monkeypatch):
    """Flip probability should compare against one random event-level draw."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))

    monkeypatch.setattr(np.random, "rand", lambda: 0.75)
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}
    result, flip_meta = FlipAugment(axis=0, p=0.5)(data, meta, ["voxels", "meta"], {})
    assert np.array_equal(result["voxels"].coords, np.asarray([[0, 1, 2], [3, 1, 2]]))
    assert flip_meta is meta

    monkeypatch.setattr(np.random, "rand", lambda: 0.25)
    tensor = make_tensor([[0, 1, 2], [3, 1, 2]], meta)
    data = {"voxels": tensor, "meta": meta}
    result, flip_meta = FlipAugment(axis=0, p=0.5)(data, meta, ["voxels", "meta"], {})
    assert np.array_equal(result["voxels"].coords, np.asarray([[3, 1, 2], [0, 1, 2]]))
    assert flip_meta is meta


def test_flip_rejects_invalid_axis():
    with pytest.raises(ValueError):
        FlipAugment(axis=3)


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan])
def test_flip_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        FlipAugment(axis=0, p=p)


def test_flip_generate_meta_snaps_to_source_grid():
    meta = make_meta(
        lower=(0.1, 0.2, 0.3),
        upper=(2.1, 4.2, 2.3),
        size=(0.5, 1.0, 0.5),
    )
    pivot = np.asarray([1.37, 1.91, 1.05], dtype=np.float32)

    flip_meta = FlipAugment(axis=0, keep_meta=False).generate_meta(meta, pivot)

    start = (flip_meta.lower - meta.lower) / flip_meta.size
    assert np.allclose(start, np.rint(start))
    assert np.allclose(
        flip_meta.upper, flip_meta.lower + flip_meta.count * flip_meta.size
    )


def test_flip_generate_meta_preserves_meta_invariant_after_float32_rounding():
    meta = make_meta(
        lower=(-389.20172, -791.87866, 12.31634),
        upper=(1110.7983, 208.12134, 84.31634),
        size=(0.75, 0.5, 0.08),
    )
    pivot = np.asarray([630.4233, -291.87866, 40.79634], dtype=np.float32)

    flip_meta = FlipAugment(axis=2, keep_meta=False).generate_meta(meta, pivot)

    assert np.allclose(
        flip_meta.upper, flip_meta.lower + flip_meta.count * flip_meta.size
    )
