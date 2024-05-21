"""Used to define which dataset entries to load at each iteration"""

import time

import numpy as np
import torch

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

__all__ = ['SequentialBatchSampler', 'RandomSequenceBatchSampler',
           'BootstrapBatchSampler']


class AbstractBatchSampler(Sampler):
    """Abstract sampler class.

    Samplers that inherit from this class should work out of the box.
    Just define the __len__ and __iter__ functions. __init__ defines
    self._num_samples and self._batch_size as well as a self._random
    RNG, if needed.
    """

    def __init__(self, dataset, batch_size, seed=None, **kwargs):
        """Check and store the values passed to the initializer,
        set the seeds appropriately.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sampler from
        batch_size : int
            Number of samples to load per iteration, per process
        seed : int, optional
            Seed to use for random sampling
        **kwargs : dict, optional
            Additional arguments to pass to the parent Sampler class
        """
        # Initialize parent class
        # TODO: `data_source` id deprecated, will need to remove with newer
        # version of pytorch
        super().__init__(data_source=None, **kwargs)

        # Initialize the random number generator with a seed
        if seed is None:
            seed = int(time.time())
        self._random = np.random.RandomState(seed=seed) # pylint: disable=E1101

        # Check that the batch_size is no larger than the number of samples
        self._batch_size = int(batch_size)
        self._num_samples = len(dataset)
        if self._batch_size < 0 or self._batch_size > self._num_samples:
            class_name = self.__class__.__name__
            raise ValueError(
                    f"{class_name} received invalid batch_size ({batch_size}) "
                    f"> num_samples ({self._num_samples}).")

        # Define the number of batches as an integer multiple of the batch size
        # to ensure that every batch has the same number of entries
        self._num_batches = self._num_samples//self._batch_size

    def __len__(self):
        """Provides the full length of the sampler.

        The length of the sampler is entirely defined by the number of
        batches, which depends on how the dataset is accessed.

        Returns
        -------
        int
            Total number of entries in the dataset
        """
        return self._num_batches*self._batch_size

    def __iter__(self):
        """Placeholder to be overridden by children classes."""
        raise NotImplementedError


class SequentialBatchSampler(AbstractBatchSampler):
    """Samples batches sequentially within the dataset."""
    name = 'sequential'

    def __iter__(self):
        """Iterates over sequential batches of data."""
        order = np.arange(len(self), dtype=np.int64)
        return iter(order)


class RandomSequenceBatchSampler(AbstractBatchSampler):
    """Samples sequential batches randomly within the dataset."""
    name = 'random_sequence'

    def __init__(self, dataset, batch_size, seed=None, **kwargs):
        """Initialize the random sequence batch sampler.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sampler from
        batch_size : int
            Number of samples to load per iteration, per process
        seed : int, optional
            Seed to use for random sampling
        **kwargs : dict, optional
            Additional arguments to pass to the parent Sampler class
        """
        # Initialize the parent class
        super().__init__(dataset, batch_size, seed, **kwargs)

        # Define the number of batches
        self._num_batches = (
                self._num_samples-self._batch_size+1)//self._batch_size
        assert self._num_batches > 0, (
                 "Not enough samples to properly randomly sample sequences. "
                f"Must provide at least {2*self._batch_size-1} samples.")

    def __iter__(self):
        """Iterates over sequential batches of data randomly located
        in the dataset.
        """
        # Pick a general offset and produce sequence starts with respect to it
        offset = self._random.randint(0, self._batch_size)
        starts = np.arange(
                offset, self._num_batches*self._batch_size,
                self._batch_size, dtype=np.int64)

        # Randomly pick the starts
        self._random.shuffle(starts)

        # Produce a sequence for each start
        batches = [np.arange(
            start, start + self._batch_size,
            dtype=np.int64) for start in starts]

        return iter(np.concatenate(batches))


class BootstrapBatchSampler(AbstractBatchSampler):
    """Sampler used for bootstrap sampling of the entire dataset.

    This is particularly useful for training an ensemble of networks
    (bagging).
    """
    name = 'bootstrap'

    def __iter__(self):
        """Iterates over bootstrapped batches of data randomly picked
        from the dataset.
        """
        max_id  = self._num_samples + 1 - self._batch_size
        starts = np.arange(
                0, max_id, self._batch_size, dtype=np.int64)
        bootstrap_indices = np.random.choice(np.arange(
            self._num_samples), self._num_samples)
        batches = [bootstrap_indices[np.arange(
            start, start+self._batch_size,
            dtype=np.int64)] for start in starts]

        return iter(np.concatenate(batches))


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        """Convert a basic sampler to an instance of a distributed sampler.

        Parameters
        ----------
        sampler : Sampler
            Input torch sampler
        num_replicas : int, optional
            Number of distributed samplers running concurrently
        rank : int, optional
            Rank of the current sampler

        Notes
        -----
        Input sampler is assumed to be of constant size.
        """
        # Initialiaze the parent distributed sampler
        super().__init__(
                sampler, num_replicas=num_replicas, rank=rank, shuffle=False)

        # Store the underlying basic sampler
        self.sampler = sampler
        self._num_samples = sampler._num_samples
        self._batch_size = sampler._batch_size

    def __iter__(self):
        """Overrides the basic iterator with one that takes into account
        the number of replicas and the rank of the sampler.
        """
        # Deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # Truncate the number of entries to be divisible by the world size
        #indices += indices[:(self.total_size - len(indices))]
        num_remove = self.total_size - self.total_size%self.num_replicas
        if num_remove > 0:
            indices = indices[:-num_remove]

        if len(indices == 0):
            raise RuntimeError(
                    f"There should be at least enough")

        if len(indices)// self.total_size:
            raise RuntimeError(f"{len(indices)} vs {self.total_size}")

        # Subsample
        batch_size = self._batch_size
        num_batches = self.total_size//batch_size
        splits = np.arange(batch_size, len(indices), batch_size)
        indices_split = np.split(indices, splits)
        indices = np.concatenate(
                indices_split[self.rank:num_batches:self.num_replicas])
        #indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self._num_samples:
            raise RuntimeError(f"{len(indices)} vs {self._num_samples}")

        return iter(indices)
