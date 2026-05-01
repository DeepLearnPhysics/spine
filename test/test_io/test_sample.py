"""Test that the collate function(s) work as intended."""

from dataclasses import dataclass

import numpy as np
import pytest

from spine.io.sample import *
from spine.io.sample import DistributedProxySampler


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
