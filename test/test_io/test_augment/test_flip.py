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
