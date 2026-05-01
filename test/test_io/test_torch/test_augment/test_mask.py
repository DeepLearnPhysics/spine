"""Tests for the mask augmenter."""

from .helpers import (
    BOX2,
    Any,
    GeoManager,
    MaskAugment,
    ParserTensor,
    make_meta,
    np,
    pytest,
)


def test_mask_accepts_use_geo_boundaries():
    """Mask should expose geometry boundary selection explicitly."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    mask = MaskAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        use_geo_boundaries=True,
    )

    assert mask.lower is not None and mask.upper is not None
    assert np.allclose(mask.lower, geo.tpc.lower)
    assert np.allclose(mask.upper, geo.tpc.upper)

    GeoManager.reset()


def test_mask_rejects_removed_use_geo_alias():
    """Mask should no longer accept the vague use_geo alias."""
    with pytest.raises(TypeError):
        mask_ctor: Any = __import__(
            "spine.io.torch.augment", fromlist=["MaskAugment"]
        ).__dict__["MaskAugment"]
        kwargs: Any = {
            "min_dimensions": BOX2,
            "max_dimensions": BOX2,
            "use_geo": True,
        }
        mask_ctor(**kwargs)


def test_mask_augment_can_bias_toward_weighted_activity_center():
    """Mask boxes should be able to target a weighted activity center."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [8, 8, 8], [8, 7, 8]], dtype=np.int64)
    features = np.asarray([[1.0], [10.0], [10.0]], dtype=np.float32)
    tensor = ParserTensor(coords=coords.copy(), features=features, meta=meta)
    data = {"voxels": tensor, "meta": meta}

    augment = MaskAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="weighted_activity",
        center_spread=np.zeros(3, dtype=np.float32),
        center_feature_index=0,
    )
    result, returned_meta = augment(data, meta, ["voxels", "meta"], {})

    assert returned_meta is meta
    assert np.array_equal(result["voxels"].coords, np.asarray([[1, 1, 1]]))


def test_mask_augment_defaults_to_weighted_activity_spread(monkeypatch):
    """Weighted activity masks should use weighted spread when none is given."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [7, 7, 7], [7, 1, 1]], dtype=np.int64)
    weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = ParserTensor(coords=coords.copy(), features=weights, meta=meta)
    data = {"voxels": tensor, "meta": meta}
    seen = {}

    def sample_box_lower(_lower, _upper, _dimensions, anchor=None, spread=None):
        seen["anchor"] = anchor
        seen["spread"] = spread
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(MaskAugment, "sample_box_lower", staticmethod(sample_box_lower))

    augment = MaskAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="weighted_activity",
        center_feature_index=0,
    )
    augment.generate_mask(data, meta, ["voxels", "meta"])

    coords_cm = meta.to_cm(coords, center=True)
    expected_center = np.average(coords_cm, axis=0, weights=weights)
    expected_spread = np.sqrt(
        np.average((coords_cm - expected_center) ** 2, axis=0, weights=weights)
    )
    assert np.allclose(seen["anchor"], expected_center)
    assert np.allclose(seen["spread"], expected_spread)


def test_mask_generate_mask_snaps_sampled_bounds_to_grid(monkeypatch):
    """Mask metadata should be aligned to the source voxel grid."""
    meta = make_meta(
        lower=(0.1, 0.2, 0.3),
        upper=(4.1, 6.2, 4.3),
        size=(0.5, 0.75, 0.5),
    )
    data = {
        "voxels": ParserTensor(
            coords=np.asarray([[1, 1, 1]], dtype=np.int64),
            features=np.asarray([[1.0]], dtype=np.float32),
            meta=meta,
        )
    }

    monkeypatch.setattr(
        MaskAugment,
        "sample_box_lower",
        staticmethod(lambda *args, **kwargs: np.asarray([1.13, 2.61, 1.44])),
    )

    augment = MaskAugment(
        min_dimensions=np.asarray([1.1, 1.6, 1.1], dtype=np.float32),
        max_dimensions=np.asarray([1.1, 1.6, 1.1], dtype=np.float32),
    )
    mask_meta = augment.generate_mask(data, meta, ["voxels"])

    start = (mask_meta.lower - meta.lower) / meta.size
    assert np.allclose(start, np.rint(start))
    assert np.allclose(mask_meta.upper, mask_meta.lower + mask_meta.count * meta.size)


def test_mask_constructor_validates_arguments():
    with pytest.raises(ValueError):
        MaskAugment(min_dimensions=BOX2)
    with pytest.raises(ValueError):
        MaskAugment()
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=np.asarray([1.0, 2.0]),
            max_dimensions=np.asarray([1.0, 2.0]),
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            lower=np.asarray([0.0, 0.0]),
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            upper=np.asarray([1.0, 1.0]),
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=np.asarray([1.0, 0.0, 1.0]),
            max_dimensions=BOX2,
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=np.asarray([3.0, 3.0, 3.0]),
            max_dimensions=BOX2,
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            lower=np.asarray([2.0, 2.0, 2.0]),
            upper=np.asarray([1.0, 3.0, 3.0]),
        )
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            use_geo_boundaries=True,
            lower=np.zeros(3, dtype=np.float32),
        )
    with pytest.raises(ValueError):
        MaskAugment(min_dimensions=BOX2, max_dimensions=BOX2, center_mode="bad")
    with pytest.raises(ValueError):
        MaskAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            center_feature_index=-1,
        )


def test_mask_generate_mask_rejects_bounds_smaller_than_range():
    meta = make_meta()
    data = {
        "voxels": ParserTensor(
            coords=np.asarray([[0, 0, 0]], dtype=np.int64),
            features=np.asarray([[1.0]], dtype=np.float32),
            meta=meta,
        )
    }
    augment = MaskAugment(
        min_dimensions=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        max_dimensions=np.asarray([5.0, 5.0, 5.0], dtype=np.float32),
        lower=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
    )

    with pytest.raises(ValueError):
        augment.generate_mask(data, meta, ["voxels"])
