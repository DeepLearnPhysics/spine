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
