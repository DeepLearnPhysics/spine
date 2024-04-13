"""Used to define which dataset entries to load at each iteration"""

import time
import numpy as np

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

__all__ = ['DistributedSampler', 'SequentialBatchSampler',
           'RandomSequenceSampler', 'BootstrapBatchSampler']


def AbstractBatchSampler(distributed=False):
    Parent = Sampler if not distributed else DistributedSampler
    class AbstractBatchSampler(Parent):
        """Abstract sampler class.

        Samplers that inherit from this class should work out of the box.
        Just define the __iter__ function. __init__ defines self._num_samples
        and self._batch_size as well as self._random RNG, if needed.
        """

        def __init__(self, dataset, batch_size, seed, rank = 0, **kwargs):
            """Check and store the values passed to the initializer,
            set the seeds appropriately.

            Parameters
            ----------
            dataset : torch.utils.data.Dataset
                Dataset to sampler from
            batch_size : int
                Number of samples to load per iteration, per process
            seed : Union[int, list]
                Seed to use for random sampling (one per process if multiple)
            rank : int, default 0
                Rank of the process executing the sampler
            **kwargs : dict, optional
                Additional arguments to pass to the parent Sampler class
            """
            # Initialize parent class
            if not distributed:
                super().__init__(**kwargs)
                self._random = np.random.RandomState(seed = seed)
            else:
                super().__init__(dataset, seed = seed, rank = rank, **kwargs)
                self._random = np.random.RandomState(seed = seed[rank])

            # Check that the number of samples is sound
            self._num_samples = len(dataset)
            if self._num_samples < 0:
                raise ValueError("%s received negative num_samples %s",
                        (self.__class__.__name__, str(num_samples)))

            # Check that the batch size is sound
            self._batch_size = int(batch_size)
            if self._batch_size < 0 \
                    or self._batch_size > self._num_samples:
                raise ValueError(
                        "%s received invalid batch_size %d for num_samples %d",
                        (self.__class__.__name__, batch_size, self._num_samples))

        def __len__(self):
            """Provides the full length of the sampler (number of entries)

            Returns
            -------
            int
                Total number of entries in the dataset
            """
            return self._num_samples

    return AbstractBatchSampler


def SequentialBatchSampler(distributed=False):
    class SequentialBatchSampler(AbstractBatchSampler(distributed)):
        """Samples batches sequentially within the dataset."""
        name = 'sequential_sampler'

        def __iter__(self):
            """Iterates over sequential batches of data."""
            num_batches = self._num_samples/self._batch_size
            order       = np.arange(
                    num_batches*self._batch_size, dtype=np.int64)

            return iter(order)

    return SequentialBatchSampler


def RandomSequenceSampler(distributed=False):
    class RandomSequenceSampler(AbstractBatchSampler(distributed)):
        """Samples sequential batches randomly within the dataset."""
        name = 'random_sequence_sampler'

        def __iter__(self):
            """Iterates over sequential batches of data randomly located
            in the dataset.
            """
            max_id  = self._num_samples + 1 - self._batch_size
            starts  = self._random.randint(0, max_id, len(self))
            batches = [np.arange(
                start, start + self._batch_size,
                dtype=np.int64) for start in starts]

            return iter(np.concatenate(batches))

    return RandomSequenceSampler


def BootstrapBatchSampler(distributed=False):
    class BootstrapBatchSampler(AbstractBatchSampler(distributed)):
        """Sampler used for bootstrap sampling of the entire dataset.

        This is particularly useful for training an ensemble of networks
        (bagging).
        """
        name = 'bootstrap_sampler'

        def __iter__(self):
            """Iterates over bootstrapped batches of data randomly picked
            from the dataset.
            """
            starts = np.arange(
                    0, self._num_samples+1 - self._batch_size, self._batch_size,
                    dtype=np.int64)
            bootstrap_indices = np.random.choice(np.arange(
                self._num_samples), self._num_samples)
            batches = [bootstrap_indices[np.arange(
                start, start+self._batch_size,
                dtype=np.int64)] for start in starts]

            return iter(np.concatenate(batches))

    return BootstrapBatchSampler
