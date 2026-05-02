"""Test that the collate function(s) work as intended."""

import importlib
from dataclasses import dataclass

import numpy as np
import pytest

import spine.io.sample as sample_module
from spine.io.sample import *
from spine.io.sample import DistributedProxySampler
from spine.utils.conditional import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch is required for sampler tests."
)


@dataclass
class DummyDataset:
    """Dataset with the basic attributes needed to tes the samplers."""

    size: int

    def __len__(self):
        """Provides the number of elements of the dataset.

        Returns
        -------
        int
            Number of elements in the dataset
        """
        return self.size


@pytest.fixture(name="dataset")
def fixture_dataset(request):
    """Generates a dummy dataset whith the necessary attributes and
    methods to test the sampler classes.
    """
    return DummyDataset(request.param)


@pytest.mark.parametrize("dataset", [12, 37], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seed", [None, 8])
def test_sequential_sampler(dataset, batch_size, seed):
    """Tests the sequential batch sampler."""
    # Initialize the sampler
    sampler = SequentialBatchSampler(dataset, batch_size, seed)

    # Check that the sampler length is as expected
    assert len(sampler) == (len(dataset) // batch_size) * batch_size

    # Check that the entire sampling list is of the expected length
    samples = np.array(list(sampler))
    assert len(samples) % batch_size == 0
    assert len(samples) == len(dataset) - len(dataset) % batch_size

    # Check that all the entries in the sampler are sequential
    assert samples[0] == 0
    assert (samples[1:] == samples[:-1] + 1).all()

    # Make sure that the sampler works in a distributed context
    if batch_size > 1:
        for rank in (0, 1):
            dist_sampler = DistributedProxySampler(sampler, num_replicas=2, rank=rank)

            # Make the distributed sampler has half the entries
            assert len(dist_sampler) == len(sampler) / 2

            # Make sure the indexes are sequentially within each minibatch
            num_batches = len(sampler) // batch_size
            mb_size = batch_size // 2
            for i in range(num_batches):
                seq = np.array(list(dist_sampler)[mb_size * i : mb_size * (i + 1)])
                assert (seq[1:] == seq[:-1] + 1).all()


@pytest.mark.parametrize("dataset", [12, 37], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seed", [None, 8])
def test_random_sequence_sampler(dataset, batch_size, seed):
    """Tests the random sequence batch sampler."""
    # Initialize the sampler
    sampler = RandomSequenceBatchSampler(
        dataset=dataset, batch_size=batch_size, seed=seed
    )

    # Check that the sampler length is as expected
    assert len(sampler) == (len(dataset) // batch_size) * batch_size

    # Check that the entire sampling list is of the expected length
    samples = np.array(list(sampler))
    offset = np.min(samples)
    assert len(samples) % batch_size == 0
    assert len(samples) == len(sampler)

    # Ensure that all indexes appear in the list
    assert (np.sort(samples) == np.arange(offset, np.max(samples) + 1)).all()

    # Check the samples are sequential within each batch
    if batch_size > 1:
        for start in np.arange(0, len(samples), batch_size):
            assert (
                samples[start + 1 : start + batch_size]
                == (samples[start : start + batch_size - 1] + 1) % len(sampler)
            ).all()

    # Make sure that the sampler works in a distributed context
    if batch_size > 1:
        for rank in (0, 1):
            dist_sampler = DistributedProxySampler(sampler, num_replicas=2, rank=rank)

            # Make the distributed sampler has half the entries
            assert len(dist_sampler) == len(sampler) / 2

            # Make sure the indexes are sequentially within each minibatch.
            # The only exception to this rule is the last batch which can wrap
            # around to the start, depending on the random offset
            num_batches = len(sampler) // batch_size
            mb_size = batch_size // 2
            for i in range(num_batches):
                seq = np.array(list(dist_sampler)[mb_size * i : mb_size * (i + 1)])
                assert (seq[1:] == (seq[:-1] + 1) % len(sampler)).all()


@pytest.mark.parametrize("dataset", [12, 37], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seed", [None, 8])
def test_bootstrap_sampler(dataset, batch_size, seed):
    """Tests the bootstrap batch sampler."""
    # Initialize the sampler
    sampler = BootstrapBatchSampler(dataset, batch_size, seed)

    # Check that the sampler length is as expected
    assert len(sampler) == (len(dataset) // batch_size) * batch_size

    # Check that the entire sampling list is of the expected length
    samples = np.array(list(sampler))
    assert len(samples) % batch_size == 0
    assert len(samples) == len(dataset) - len(dataset) % batch_size

    # Make sure that the sampler works in a distributed context
    if batch_size > 1:
        for rank in (0, 1):
            dist_sampler = DistributedProxySampler(sampler, num_replicas=2, rank=rank)

            # Make the distributed sampler has half the entries
            assert len(dist_sampler) == len(sampler) / 2


def test_sampler_input_validation():
    """Sampler base classes should reject invalid construction parameters."""
    dataset = DummyDataset(4)

    with pytest.raises(AssertionError, match="must be an integer"):
        SequentialBatchSampler(dataset, batch_size=2, seed="bad")
    with pytest.raises(ValueError, match="positive non-zero integer"):
        SequentialBatchSampler(dataset, batch_size=0)
    with pytest.raises(ValueError, match="at least on entry"):
        SequentialBatchSampler(DummyDataset(0), batch_size=1)
    with pytest.raises(ValueError, match="does not have enough samples"):
        SequentialBatchSampler(DummyDataset(1), batch_size=2)


def test_sampler_without_drop_last():
    """Samplers should preserve trailing entries when drop_last is disabled."""
    dataset = DummyDataset(5)
    sampler = SequentialBatchSampler(dataset, batch_size=2, drop_last=False, seed=1)
    assert len(sampler) == 5
    assert np.array_equal(np.array(list(sampler)), np.arange(5))


def test_abstract_sampler_iter_not_implemented():
    """The abstract sampler iterator should remain abstract in practice."""
    sampler = sample_module.AbstractBatchSampler(DummyDataset(1), batch_size=1, seed=1)
    with pytest.raises(NotImplementedError):
        iter(sampler).__next__()


def test_distributed_proxy_sampler_requires_divisible_batch():
    """Distributed sampling should reject incompatible batch sizes."""
    sampler = SequentialBatchSampler(DummyDataset(4), batch_size=3, seed=1)
    with pytest.raises(AssertionError, match="must be a multiple"):
        DistributedProxySampler(sampler, num_replicas=2, rank=0)


def test_sample_module_import_safe_without_torch(monkeypatch):
    """The sampler module should expose stand-ins when torch is unavailable."""
    import spine.utils.conditional as conditional

    monkeypatch.setattr(conditional, "TORCH_AVAILABLE", False)
    reloaded = importlib.reload(sample_module)
    try:
        reloaded.Sampler()
        assert reloaded.Sampler is not None
        with pytest.raises(ImportError, match="PyTorch is required"):
            reloaded.DistributedSampler()
    finally:
        monkeypatch.setattr(conditional, "TORCH_AVAILABLE", TORCH_AVAILABLE)
        importlib.reload(sample_module)


def test_distributed_proxy_sampler_pads_when_padding_exceeds_length():
    """Distributed sampling should pad with repeated indices when needed."""
    sampler = SequentialBatchSampler(
        DummyDataset(1), batch_size=5, drop_last=False, seed=1
    )
    dist_sampler = DistributedProxySampler(sampler, num_replicas=5, rank=0)
    assert list(dist_sampler) == [0]


def test_distributed_proxy_sampler_pads_with_prefix_copy():
    """Distributed sampling should reuse the prefix when only a small pad is needed."""
    sampler = SequentialBatchSampler(
        DummyDataset(5), batch_size=4, drop_last=False, seed=1
    )
    dist_sampler = DistributedProxySampler(sampler, num_replicas=4, rank=0)

    assert list(dist_sampler) == [0, 4]
