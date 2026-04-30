"""Tests for the crop augmenter."""

from .helpers import (
    BOX2,
    Any,
    CropAugment,
    GeoManager,
    ParserTensor,
    make_meta,
    make_tensor,
    np,
    pytest,
)


def test_crop_augment_can_bias_toward_activity_center():
    """Crop boxes should be able to lock onto the activity center."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    tensor = make_tensor([[7, 7, 7], [8, 7, 7], [7, 8, 7]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="activity",
        center_spread=np.zeros(3, dtype=np.float32),
        keep_meta=False,
    )
    result, crop_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.allclose(crop_meta.lower, np.asarray([6.8333335, 6.8333335, 6.5]))
    assert np.allclose(crop_meta.upper, np.asarray([8.8333335, 8.8333335, 8.5]))
    assert np.array_equal(
        result["voxels"].coords, np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    )


def test_crop_augment_can_keep_meta_fixed():
    """Cropping should optionally preserve the original metadata and indices."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    tensor = make_tensor([[1, 1, 1], [7, 7, 7], [8, 7, 7], [7, 8, 7]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        lower=np.asarray([7.0, 7.0, 7.0], dtype=np.float32),
        upper=np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        keep_meta=True,
    )
    result, crop_meta = augment(data, meta, ["voxels", "meta"], {})

    assert crop_meta is meta
    assert np.array_equal(
        result["voxels"].coords, np.asarray([[7, 7, 7], [8, 7, 7], [7, 8, 7]])
    )
    assert np.array_equal(
        result["voxels"].features, np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    )


def test_crop_accepts_use_geo_boundaries():
    """Crop should expose geometry boundary selection explicitly."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    crop = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        use_geo_boundaries=True,
    )

    assert crop.lower is not None and crop.upper is not None
    assert np.allclose(crop.lower, geo.tpc.lower)
    assert np.allclose(crop.upper, geo.tpc.upper)

    GeoManager.reset()


def test_crop_rejects_removed_use_geo_alias():
    """Crop should no longer accept the vague use_geo alias."""
    with pytest.raises(TypeError):
        crop_ctor: Any = __import__(
            "spine.io.torch.augment", fromlist=["CropAugment"]
        ).__dict__["CropAugment"]
        kwargs: Any = {
            "min_dimensions": BOX2,
            "max_dimensions": BOX2,
            "use_geo": True,
        }
        crop_ctor(**kwargs)


def test_crop_augment_can_drop_points_outside_active_volume():
    """Cropping should be able to clip points to detector module active volumes."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    lower = np.floor(geo.tpc.lower) - 1.0
    upper = np.ceil(geo.tpc.upper) + 2.0
    meta = make_meta(lower=lower, upper=upper)
    inside_cm = geo.tpc.modules[0].center.reshape(1, -1)
    outside_cm = (geo.tpc.upper + 0.5).reshape(1, -1)
    coords = np.rint(meta.to_px(np.vstack([inside_cm, outside_cm])) - 0.5).astype(
        np.int64
    )
    tensor = ParserTensor(
        coords=coords.copy(),
        features=np.asarray([[1.0], [2.0]], dtype=np.float32),
        meta=meta,
    )
    data = {"voxels": tensor, "meta": meta}

    augment = CropAugment(active_volume=True)
    result, returned_meta = augment(data, meta, ["voxels", "meta"], {})

    assert np.array_equal(
        result["voxels"].coords,
        returned_meta.to_px(inside_cm, floor=True).astype(np.int64),
    )
    assert np.array_equal(
        result["voxels"].features, np.asarray([[1.0]], dtype=np.float32)
    )
    assert np.all(returned_meta.lower >= meta.lower)
    assert np.all(returned_meta.upper <= meta.upper)

    GeoManager.reset()


def test_crop_active_volume_can_keep_meta_fixed():
    """Active-volume crop should optionally preserve the original metadata."""
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    lower = np.floor(geo.tpc.lower) - 1.0
    upper = np.ceil(geo.tpc.upper) + 2.0
    meta = make_meta(lower=lower, upper=upper)
    inside_cm = geo.tpc.modules[0].center.reshape(1, -1)
    outside_cm = (geo.tpc.upper + 0.5).reshape(1, -1)
    coords = np.rint(meta.to_px(np.vstack([inside_cm, outside_cm])) - 0.5).astype(
        np.int64
    )
    tensor = ParserTensor(
        coords=coords.copy(),
        features=np.asarray([[1.0], [2.0]], dtype=np.float32),
        meta=meta,
    )
    data = {"voxels": tensor, "meta": meta}

    augment = CropAugment(active_volume=True, keep_meta=True)
    result, returned_meta = augment(data, meta, ["voxels", "meta"], {})

    assert returned_meta is meta
    assert np.array_equal(result["voxels"].coords, coords[[0]])
    assert np.array_equal(
        result["voxels"].features, np.asarray([[1.0]], dtype=np.float32)
    )

    GeoManager.reset()


def test_crop_constructor_validates_arguments():
    with pytest.raises(ValueError):
        CropAugment(min_dimensions=BOX2)
    with pytest.raises(ValueError):
        CropAugment()
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=np.asarray([1.0, 2.0]),
            max_dimensions=np.asarray([1.0, 2.0]),
        )
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            lower=np.asarray([0.0, 0.0]),
        )
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            upper=np.asarray([1.0, 1.0]),
        )
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=np.asarray([1.0, 0.0, 1.0]),
            max_dimensions=BOX2,
        )
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=np.asarray([3.0, 3.0, 3.0]),
            max_dimensions=BOX2,
        )
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            lower=np.asarray([2.0, 2.0, 2.0]),
            upper=np.asarray([1.0, 3.0, 3.0]),
        )
    with pytest.raises(ValueError):
        CropAugment(active_volume=True, use_geo_boundaries=True)
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            use_geo_boundaries=True,
            lower=np.zeros(3, dtype=np.float32),
        )
    with pytest.raises(ValueError):
        CropAugment(min_dimensions=BOX2, max_dimensions=BOX2, center_mode="bad")
    with pytest.raises(ValueError):
        CropAugment(
            min_dimensions=BOX2,
            max_dimensions=BOX2,
            center_feature_index=-1,
        )


def test_crop_generate_crop_validates_internal_configuration():
    meta = make_meta()
    data = {"voxels": make_tensor([[0, 0, 0]], meta), "meta": meta}

    augment = CropAugment(active_volume=True)
    with pytest.raises(ValueError):
        augment.generate_crop(data, meta, ["voxels"])

    too_small = CropAugment(
        min_dimensions=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        max_dimensions=np.asarray([5.0, 5.0, 5.0], dtype=np.float32),
        lower=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        too_small.generate_crop(data, meta, ["voxels"])


def test_crop_apply_rejects_missing_output_meta_in_inconsistent_state():
    meta = make_meta()
    data = {"voxels": make_tensor([[0, 0, 0]], meta), "meta": meta}
    augment = CropAugment(active_volume=True, keep_meta=False)

    augment.active_volume = False
    with pytest.raises(ValueError):
        augment(data, meta, ["voxels", "meta"], {})
