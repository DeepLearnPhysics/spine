"""Tests for the crop augmenter."""

from .helpers import (
    BOX2,
    Any,
    CropAugment,
    GeoManager,
    TensorData,
    make_meta,
    make_tensor,
    np,
    pytest,
)


def test_crop_augment_can_bias_toward_activity_center():
    """Crop boxes should bias toward activity while staying on the source grid."""
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

    assert np.allclose(crop_meta.lower, np.asarray([7.0, 7.0, 6.0]))
    assert np.allclose(crop_meta.upper, np.asarray([9.0, 9.0, 8.0]))
    assert np.array_equal(
        result["voxels"].coords, np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    )


def test_crop_augment_defaults_to_activity_spread(monkeypatch):
    """Activity-biased crops should use activity spread when none is given."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [7, 7, 7], [7, 1, 1]], dtype=np.int64)
    tensor = TensorData(
        coords=coords,
        features=np.ones((3, 1), dtype=np.float32),
        meta=meta,
    )
    data = {"voxels": tensor, "meta": meta}
    seen = {}

    def sample_box_lower(_lower, _upper, _dimensions, anchor=None, spread=None):
        seen["anchor"] = anchor
        seen["spread"] = spread
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(CropAugment, "sample_box_lower", staticmethod(sample_box_lower))

    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="activity",
        keep_meta=False,
    )
    augment.generate_crop(data, meta, ["voxels", "meta"])

    coords_cm = meta.to_cm(coords, center=True)
    assert np.allclose(seen["anchor"], np.mean(coords_cm, axis=0))
    assert np.allclose(seen["spread"], np.std(coords_cm, axis=0))


def test_crop_activity_center_can_use_one_explicit_product(monkeypatch):
    """An explicit center key excludes unrelated coordinate products."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    image = make_tensor([[1, 1, 1], [2, 1, 1]], meta)
    labels = make_tensor([[8, 8, 8]], meta)
    data = {"data": image, "labels": labels, "meta": meta}
    seen = {}

    def sample_box_lower(_lower, _upper, _dimensions, anchor=None, spread=None):
        seen["anchor"] = anchor
        return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(
        CropAugment,
        "sample_box_lower",
        staticmethod(sample_box_lower),
    )
    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="activity",
        center_key="data",
    )

    augment.generate_crop(data, meta, list(data))

    expected = np.mean(meta.to_cm(image.coords, center=True), axis=0)
    assert np.allclose(seen["anchor"], expected)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"missing": TensorData(np.ones((1, 1), dtype=np.float32), feats_only=True)},
    ],
)
def test_crop_activity_center_rejects_invalid_explicit_product(data):
    """An explicit center source must carry coordinates."""
    meta = make_meta()
    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        center_mode="activity",
        center_key="missing",
    )

    with pytest.raises(ValueError, match="center_key"):
        augment.generate_crop(data, meta, [])


def test_crop_generate_crop_snaps_sampled_bounds_to_grid(monkeypatch):
    """Crop metadata should be aligned to the source voxel grid."""
    meta = make_meta(
        lower=(0.1, 0.2, 0.3),
        upper=(4.1, 6.2, 4.3),
        size=(0.5, 0.75, 0.5),
    )
    data = {"voxels": make_tensor([[1, 1, 1]], meta), "meta": meta}

    monkeypatch.setattr(
        CropAugment,
        "sample_box_lower",
        staticmethod(lambda *args, **kwargs: np.asarray([1.13, 2.61, 1.44])),
    )

    augment = CropAugment(
        min_dimensions=np.asarray([1.1, 1.6, 1.1], dtype=np.float32),
        max_dimensions=np.asarray([1.1, 1.6, 1.1], dtype=np.float32),
        keep_meta=False,
    )
    crop_meta = augment.generate_crop(data, meta, ["voxels", "meta"])

    start = (crop_meta.lower - meta.lower) / meta.size
    assert np.allclose(start, np.rint(start))
    assert np.allclose(crop_meta.upper, crop_meta.lower + crop_meta.count * meta.size)


def test_crop_generate_crop_preserves_meta_invariant_after_float32_rounding(
    monkeypatch,
):
    """Crop metadata should remain valid after float32 storage rounding."""
    meta = make_meta(
        lower=(-389.20172, -791.87866, 12.31634),
        upper=(1110.7983, 208.12134, 84.31634),
        size=(0.75, 0.5, 0.08),
    )
    data = {"voxels": make_tensor([[0, 0, 0]], meta), "meta": meta}

    monkeypatch.setattr(
        CropAugment,
        "sample_box_lower",
        staticmethod(lambda *args, **kwargs: np.asarray([627.65, 172.0, -2.7])),
    )

    augment = CropAugment(
        min_dimensions=np.asarray([5.25, 21.0, 2.64], dtype=np.float32),
        max_dimensions=np.asarray([5.25, 21.0, 2.64], dtype=np.float32),
        keep_meta=False,
    )
    crop_meta = augment.generate_crop(data, meta, ["voxels", "meta"])

    assert np.allclose(crop_meta.upper, crop_meta.lower + crop_meta.count * meta.size)


def test_crop_rejects_maximum_size_that_cannot_fit():
    """Variable crop configurations fail deterministically when their max is invalid."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    augment = CropAugment(
        min_dimensions=np.full(3, 2.0, dtype=np.float32),
        max_dimensions=np.full(3, 5.0, dtype=np.float32),
    )

    for seed in range(5):
        np.random.seed(seed)
        with pytest.raises(ValueError, match="maximum cropping dimensions"):
            augment.generate_crop({}, meta, [])


def test_crop_intersects_configured_bounds_with_input_image(monkeypatch):
    """Sampling bounds cannot extend a crop beyond the available source image."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    seen = {}

    def sample_box_lower(lower, upper, _dimensions, **_kwargs):
        seen["lower"] = lower
        seen["upper"] = upper
        return lower

    monkeypatch.setattr(
        CropAugment,
        "sample_box_lower",
        staticmethod(sample_box_lower),
    )
    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        lower=np.full(3, -5.0, dtype=np.float32),
        upper=np.full(3, 15.0, dtype=np.float32),
    )

    crop_meta = augment.generate_crop({}, meta, [])

    assert np.array_equal(seen["lower"], meta.lower)
    assert np.array_equal(seen["upper"], meta.upper)
    assert np.all(crop_meta.lower >= meta.lower)
    assert np.all(crop_meta.upper <= meta.upper)


def test_crop_rejects_bounds_outside_input_image():
    """Configured sampling bounds must overlap the source image."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        lower=np.full(3, 20.0, dtype=np.float32),
        upper=np.full(3, 30.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="do not overlap"):
        augment.generate_crop({}, meta, [])


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


def test_crop_preserves_continuous_point_coordinates(monkeypatch):
    """Continuous truth points are selected and rebased without quantization."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    crop_meta = make_meta(lower=(2.0, 2.0, 2.0), upper=(4.0, 4.0, 4.0))
    voxels = make_tensor([[2, 2, 2], [4, 4, 4]], meta)
    points = TensorData(
        coords=np.asarray(
            [[1.75, 2.25, 2.25], [2.25, 2.25, 2.25], [3.75, 3.75, 3.75]],
            dtype=np.float32,
        ),
        features=np.arange(3, dtype=np.float32).reshape(-1, 1),
        meta=meta,
    )
    data = {"voxels": voxels, "points": points, "meta": meta}
    augment = CropAugment(
        min_dimensions=BOX2,
        max_dimensions=BOX2,
        keep_meta=False,
    )
    monkeypatch.setattr(augment, "generate_crop", lambda *_args: crop_meta)

    result, output_meta = augment(data, meta, list(data), {})

    assert output_meta is crop_meta
    assert np.array_equal(result["voxels"].coords, np.asarray([[0, 0, 0]]))
    assert np.allclose(
        result["points"].coords,
        np.asarray([[0.25, 0.25, 0.25], [1.75, 1.75, 1.75]]),
    )
    assert np.array_equal(
        result["points"].features,
        np.asarray([[1.0], [2.0]], dtype=np.float32),
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
            "spine.io.augment", fromlist=["CropAugment"]
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
    tensor = TensorData(
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
    tensor = TensorData(
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
            center_key="data",
        )
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
