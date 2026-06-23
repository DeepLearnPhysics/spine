"""Tests for the translate augmenter."""

from spine.io.augment.translate import TranslateAugment

from .helpers import GeoManager, make_meta, make_tensor, np, pytest


def test_translate_rejects_bad_bounds_and_conflicts():
    with pytest.raises(ValueError):
        TranslateAugment(lower=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        TranslateAugment(
            lower=np.zeros(2, dtype=np.float32), upper=np.ones(2, dtype=np.float32)
        )
    with pytest.raises(ValueError):
        TranslateAugment(
            lower=np.ones(3, dtype=np.float32), upper=np.zeros(3, dtype=np.float32)
        )

    GeoManager.reset()
    GeoManager.initialize_or_get(detector="icarus")
    with pytest.raises(ValueError):
        TranslateAugment(
            lower=np.zeros(3, dtype=np.float32),
            upper=np.ones(3, dtype=np.float32),
            use_geo=True,
        )
    GeoManager.reset()


def test_translate_uses_original_meta_when_no_target_given():
    meta = make_meta()
    other = make_meta(lower=(1.0, 1.0, 1.0), upper=(3.0, 5.0, 3.0))
    augment = TranslateAugment()
    target = augment.get_target_meta(other, meta)
    assert np.array_equal(target.lower, meta.lower)
    assert np.array_equal(target.upper, meta.upper)


def test_translate_uses_custom_meta_and_fills_pitch_when_missing():
    augment = TranslateAugment(
        lower=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.asarray([6.0, 6.0, 6.0], dtype=np.float32),
    )
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0), size=(2.0, 2.0, 2.0))
    tensor = make_tensor([[0, 0, 0]], meta)
    data = {"voxels": tensor, "meta": meta}

    original = augment.generate_offset
    augment.generate_offset = lambda meta_arg, target_arg: np.asarray(
        [1, 1, 1], dtype=np.int64
    )
    result, target_meta = augment(
        data, meta, ["voxels", "meta"], {"original_meta": meta}
    )
    augment.generate_offset = original

    assert np.array_equal(target_meta.size, meta.size)
    assert np.array_equal(target_meta.count, np.asarray([3, 3, 3]))
    assert np.array_equal(result["voxels"].coords, np.asarray([[1, 1, 1]]))


def test_translate_supports_geo_target_meta():
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")
    augment = TranslateAugment(use_geo=True)
    target = augment.get_target_meta(make_meta())
    assert np.allclose(target.lower, geo.tpc.lower)
    assert np.all(target.upper >= geo.tpc.upper)
    GeoManager.reset()


def test_translate_generate_offset_checks_pixel_pitch():
    meta = make_meta(size=(1.0, 1.0, 1.0))
    target = make_meta(size=(2.0, 1.0, 1.0))
    augment = TranslateAugment()
    with pytest.raises(ValueError):
        augment.generate_offset(meta, target)


def test_translate_generate_offset_stays_within_available_range():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(2.0, 2.0, 2.0))
    target = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    augment = TranslateAugment()
    offset = augment.generate_offset(meta, target)
    assert offset.shape == (3,)
    assert np.all(offset >= 0)
    assert np.all(offset <= (target.count - meta.count))
