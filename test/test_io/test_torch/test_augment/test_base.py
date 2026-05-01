"""Tests for shared augmenter helpers."""

from spine.io.torch.augment.base import AugmentBase

from .helpers import BOX2, GeoManager, ParserTensor, make_meta, np, pytest


class DummyAugment(AugmentBase):
    """Minimal augment to expose base helpers."""

    name = "dummy"

    def apply(self, data, meta, keys, context):
        return data, meta


def test_resolve_center_rejects_conflicting_sources():
    meta = make_meta()
    with pytest.raises(ValueError):
        DummyAugment.resolve_center(
            meta, center=np.zeros(3, dtype=np.float32), use_geo_center=True
        )


def test_resolve_center_rejects_bad_shape():
    meta = make_meta()
    with pytest.raises(ValueError):
        DummyAugment.resolve_center(meta, center=np.zeros(2, dtype=np.float32))


def test_resolve_center_uses_geometry_center():
    GeoManager.reset()
    geo = GeoManager.initialize_or_get(detector="icarus")

    meta = make_meta()
    center = DummyAugment.resolve_center(meta, use_geo_center=True)

    assert np.allclose(center, geo.tpc.center)
    GeoManager.reset()


def test_parse_optional_vector_handles_none_scalar_and_vector():
    assert DummyAugment.parse_optional_vector(None, "foo") is None
    assert np.array_equal(
        DummyAugment.parse_optional_vector(1.5, "foo"),
        np.asarray([1.5, 1.5, 1.5], dtype=np.float32),
    )
    assert np.array_equal(
        DummyAugment.parse_optional_vector(np.asarray([1.0, 2.0, 3.0]), "foo"),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_parse_optional_vector_rejects_bad_shape():
    with pytest.raises(ValueError):
        DummyAugment.parse_optional_vector(np.asarray([1.0, 2.0]), "foo")


def test_resolve_activity_center_falls_back_to_meta_center():
    meta = make_meta(lower=(1.0, 2.0, 3.0), upper=(5.0, 6.0, 7.0))
    center = DummyAugment.resolve_activity_center({"meta": meta}, ["meta"], meta)
    assert np.allclose(center, np.asarray([3.0, 4.0, 5.0], dtype=np.float32))


def test_resolve_activity_center_handles_zero_weight_sum():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [7, 7, 7]], dtype=np.int64)
    features = np.zeros((2, 1), dtype=np.float32)
    tensor = ParserTensor(coords=coords, features=features, meta=meta)

    center = DummyAugment.resolve_activity_center(
        {"voxels": tensor}, ["voxels"], meta, weighted=True, feature_index=0
    )
    expected = np.mean(meta.to_cm(coords, center=True), axis=0)
    assert np.allclose(center, expected)


def test_resolve_activity_center_skips_empty_tensors_and_supports_1d_weights():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    empty = ParserTensor(
        coords=np.empty((0, 3), dtype=np.int64),
        features=np.empty((0,), dtype=np.float32),
        meta=meta,
    )
    coords = np.asarray([[1, 1, 1], [8, 8, 8]], dtype=np.int64)
    tensor = ParserTensor(
        coords=coords,
        features=np.asarray([1.0, 3.0], dtype=np.float32),
        meta=meta,
    )

    center = DummyAugment.resolve_activity_center(
        {"empty": empty, "voxels": tensor},
        ["empty", "voxels"],
        meta,
        weighted=True,
        feature_index=99,
    )

    coords_cm = meta.to_cm(coords, center=True)
    expected = np.average(coords_cm, axis=0, weights=np.asarray([1.0, 3.0]))
    assert np.allclose(center, expected)


def test_resolve_activity_stats_returns_activity_spread():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [7, 7, 7], [7, 1, 1]], dtype=np.int64)
    tensor = ParserTensor(
        coords=coords,
        features=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        meta=meta,
    )

    center, spread = DummyAugment.resolve_activity_stats(
        {"voxels": tensor}, ["voxels"], meta
    )

    coords_cm = meta.to_cm(coords, center=True)
    assert np.allclose(center, np.mean(coords_cm, axis=0))
    assert np.allclose(spread, np.std(coords_cm, axis=0))


def test_resolve_activity_stats_returns_weighted_activity_spread():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(10.0, 10.0, 10.0))
    coords = np.asarray([[1, 1, 1], [7, 7, 7], [7, 1, 1]], dtype=np.int64)
    weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = ParserTensor(
        coords=coords,
        features=weights,
        meta=meta,
    )

    center, spread = DummyAugment.resolve_activity_stats(
        {"voxels": tensor}, ["voxels"], meta, weighted=True
    )

    coords_cm = meta.to_cm(coords, center=True)
    expected_center = np.average(coords_cm, axis=0, weights=weights)
    expected_spread = np.sqrt(
        np.average((coords_cm - expected_center) ** 2, axis=0, weights=weights)
    )
    assert np.allclose(center, expected_center)
    assert np.allclose(spread, expected_spread)


def test_sample_box_lower_supports_uniform_and_anchored_modes():
    lower = np.zeros(3, dtype=np.float32)
    upper = np.full(3, 10.0, dtype=np.float32)
    dimensions = BOX2.copy()

    np.random.seed(1)
    sampled_uniform = DummyAugment.sample_box_lower(lower, upper, dimensions)
    assert sampled_uniform.shape == (3,)
    assert np.all(sampled_uniform >= lower)
    assert np.all(sampled_uniform <= upper - dimensions)

    sampled_anchor = DummyAugment.sample_box_lower(
        lower,
        upper,
        dimensions,
        anchor=np.asarray([7.0, 7.0, 7.0], dtype=np.float32),
        spread=np.zeros(3, dtype=np.float32),
    )
    assert np.allclose(sampled_anchor, np.asarray([6.0, 6.0, 6.0], dtype=np.float32))


def test_sample_box_lower_defaults_spread_and_clips_anchor():
    lower = np.zeros(3, dtype=np.float32)
    upper = np.full(3, 10.0, dtype=np.float32)
    dimensions = BOX2.copy()

    np.random.seed(2)
    sampled = DummyAugment.sample_box_lower(
        lower,
        upper,
        dimensions,
        anchor=np.asarray([50.0, -10.0, 5.0], dtype=np.float32),
    )

    assert np.all(sampled >= lower)
    assert np.all(sampled <= upper - dimensions)
