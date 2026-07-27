"""Tests for the jitter augmenter."""

from spine.io.augment.jitter import JitterAugment

from .helpers import make_meta, make_tensor, np, pytest


def test_jitter_rejects_bad_arguments():
    with pytest.raises(ValueError):
        JitterAugment(max_offset=np.asarray([1, 2]))
    with pytest.raises(ValueError):
        JitterAugment(max_offset=np.asarray([1, -1, 0]))
    with pytest.raises(ValueError):
        JitterAugment(max_offset=1, distribution="gaussian")
    with pytest.raises(ValueError):
        JitterAugment(max_offset=1, poisson_lambda=np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError):
        JitterAugment(max_offset=1, poisson_lambda=np.asarray([1.0, -1.0, 0.0]))


def test_jitter_apply_clips_by_default():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 0, 0], [3, 3, 3]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = JitterAugment(max_offset=np.asarray([2, 2, 2]), clip=True)
    original = augment.generate_offsets
    augment.generate_offsets = lambda n: np.asarray(
        [[-2, -2, -2], [2, 2, 2]], dtype=np.int64
    )
    result, returned_meta = augment(data, meta, ["voxels", "meta"], {})
    augment.generate_offsets = original

    assert returned_meta is meta
    assert np.array_equal(result["voxels"].coords, np.asarray([[0, 0, 0], [3, 3, 3]]))


def test_jitter_apply_can_skip_clipping():
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 0, 0]], meta)
    data = {"voxels": tensor, "meta": meta}

    augment = JitterAugment(max_offset=1, clip=False)
    original = augment.generate_offsets
    augment.generate_offsets = lambda n: np.asarray([[-1, 2, 0]], dtype=np.int64)
    result, _ = augment(data, meta, ["voxels", "meta"], {})
    augment.generate_offsets = original

    assert np.array_equal(result["voxels"].coords, np.asarray([[-1, 2, 0]]))


def test_jitter_accepts_metadata_without_coordinate_products():
    """Metadata-only events pass through without sampling offsets."""
    meta = make_meta()
    data = {"meta": meta}
    augment = JitterAugment(max_offset=1)

    result, returned_meta = augment(data, meta, ["meta"], {})

    assert result is data
    assert returned_meta is meta


def test_jitter_shares_offsets_across_reordered_product_subsets():
    """A physical coordinate receives the same offset in every product."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0))
    first = make_tensor([[0, 0, 0], [1, 0, 0]], meta)
    second = make_tensor([[2, 0, 0], [0, 0, 0], [3, 0, 0]], meta)
    second.coords = second.coords.astype(np.int32)
    data = {"first": first, "second": second}

    augment = JitterAugment(max_offset=1, clip=False)
    original = augment.generate_offsets
    augment.generate_offsets = lambda n: np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]],
        dtype=np.int64,
    )
    result, _ = augment(data, meta, ["first", "second"], {})
    augment.generate_offsets = original

    assert np.array_equal(
        result["first"].coords,
        np.asarray([[1, 0, 0], [1, 1, 0]]),
    )
    assert np.array_equal(
        result["second"].coords,
        np.asarray([[2, 0, 1], [1, 0, 0], [2, 0, 0]]),
    )
    assert result["second"].coords.dtype == np.int32


def test_jitter_samples_once_per_unique_coordinate():
    """Duplicate source rows remain aligned and retain their multiplicity."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0))
    tensor = make_tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], meta)
    original_features = tensor.features.copy()
    data = {"voxels": tensor}

    sampled_sizes = []
    augment = JitterAugment(max_offset=1, clip=False)
    original = augment.generate_offsets

    def generate_offsets(num_voxels):
        sampled_sizes.append(num_voxels)
        return np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int64)

    augment.generate_offsets = generate_offsets
    result, _ = augment(data, meta, ["voxels"], {})
    augment.generate_offsets = original

    assert sampled_sizes == [2]
    assert np.array_equal(
        result["voxels"].coords,
        np.asarray([[2, 1, 1], [2, 1, 1], [2, 3, 2]]),
    )
    assert np.array_equal(result["voxels"].features, original_features)


def test_jitter_retains_rows_when_distinct_coordinates_collide():
    """Collision reduction belongs to downstream sparse consumers."""
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0))
    tensor = make_tensor([[0, 0, 0], [1, 0, 0]], meta)
    original_features = tensor.features.copy()
    data = {"voxels": tensor}

    augment = JitterAugment(max_offset=1, clip=False)
    original = augment.generate_offsets
    augment.generate_offsets = lambda n: np.asarray(
        [[1, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )
    result, _ = augment(data, meta, ["voxels"], {})
    augment.generate_offsets = original

    assert np.array_equal(
        result["voxels"].coords,
        np.asarray([[1, 0, 0], [1, 0, 0]]),
    )
    assert np.array_equal(result["voxels"].features, original_features)
    assert len(result["voxels"].coords) == 2


def test_jitter_generate_offsets_supports_uniform_and_poisson():
    uniform = JitterAugment(max_offset=1)
    offsets = uniform.generate_offsets(5)
    assert offsets.shape == (5, 3)
    assert np.all(offsets >= -1)
    assert np.all(offsets <= 1)

    poisson = JitterAugment(
        max_offset=np.asarray([1, 2, 3]),
        distribution="poisson",
        poisson_lambda=np.asarray([10.0, 10.0, 10.0]),
    )
    offsets = poisson.generate_poisson_offsets(20)
    assert offsets.shape == (20, 3)
    assert np.all(np.abs(offsets) <= np.asarray([1, 2, 3]))


def test_jitter_generate_offsets_dispatches_to_poisson_sampler():
    augment = JitterAugment(max_offset=1, distribution="poisson")
    original = augment.generate_poisson_offsets
    augment.generate_poisson_offsets = lambda n: np.full((n, 3), 7, dtype=np.int64)
    offsets = augment.generate_offsets(2)
    augment.generate_poisson_offsets = original

    assert np.array_equal(offsets, np.full((2, 3), 7, dtype=np.int64))
