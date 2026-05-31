"""Used to define which dataset entries to load at each iteration"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sized
from typing import Any

import numpy as np

from spine.utils.conditional import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from torch.utils.data import Sampler
    from torch.utils.data.distributed import DistributedSampler
else:

    class Sampler:
        """Import-safe stand-in used when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

    class DistributedSampler:
        """Import-safe stand-in used when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for distributed sampling.")


__all__ = [
    "SequentialBatchSampler",
    "RandomSequenceBatchSampler",
    "BootstrapBatchSampler",
    "JointSequentialBatchSampler",
    "JointRandomSequenceBatchSampler",
    "JointBootstrapBatchSampler",
]


class AbstractBatchSampler(Sampler):
    """Abstract sampler class.

    Samplers that inherit from this class should work out of the box.
    Just define the __len__ and __iter__ functions. __init__ defines
    self.num_samples and self.batch_size as well as a self._random
    RNG, if needed.
    """

    joint = False

    def __init__(
        self,
        dataset: Sized,
        batch_size: int,
        seed: int | None = None,
        drop_last: bool = True,
    ) -> None:
        """Check and store the values passed to the initializer,
        set the seeds appropriately.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sampler from
        batch_size : int
            Number of samples to load per iteration
        seed : int, optional
            Seed to use for random sampling
        drop_last: bool, default True
            If `True`, drop the last batch to make the number of entries a
            multiple of the batch_size (if needed)
        """
        # Initialize parent class
        # TODO: `data_source` id deprecated, will need to remove with newer
        # version of pytorch (2.2.0 and above)
        super().__init__(data_source=None)

        # Initialize the random number generator with a seed
        if seed is None:
            seed = int(time.time())
        else:
            assert isinstance(
                seed, int
            ), f"The sampler seed must be an integer, got: {seed}."

        self._random = np.random.RandomState(seed=seed)  # pylint: disable=E1101

        # Check that the batch_size is a sensible value
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("The `batch_size` must be a positive non-zero integer.")

        # Process the number of samples
        self.num_samples = len(dataset)
        self.drop_last = drop_last
        if self.num_samples < 1:
            raise ValueError("The dataset must have at least on entry to sample from.")
        if drop_last:
            if self.num_samples < self.batch_size:
                raise ValueError(
                    "The dataset does not have enough samples "
                    f"({self.num_samples}) to produce a complete batch "
                    f"({batch_size})."
                )
            self.num_samples -= self.num_samples % self.batch_size

    def __len__(self) -> int:
        """Provides the full length of the sampler.

        The length of the sampler can differ from the number of elements in
        the underlying dataset, if the last batch is smaller than the requested
        size and is dropped.

        Returns
        -------
        int
            Total number of entries to sample
        """
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        """Placeholder to be overridden by children classes."""
        raise NotImplementedError


class SequentialBatchSampler(AbstractBatchSampler):
    """Samples batches sequentially within the dataset."""

    name = "sequential"

    def __iter__(self) -> Iterator[int]:
        """Iterates over sequential batches of data."""
        order = np.arange(self.num_samples, dtype=int)
        return iter(order)


class RandomSequenceBatchSampler(AbstractBatchSampler):
    """Samples sequential batches randomly within the dataset."""

    name = "random_sequence"

    def __iter__(self) -> Iterator[int]:
        """Iterates over sequential batches of data randomly located
        in the dataset.
        """
        # Pick a general offset and produce sequence starts with respect to it.
        # Introducing this random offset ensures that the data is not
        # systematically batched in the same way
        offset = self._random.randint(0, self.batch_size)
        starts = np.arange(
            offset, offset + self.num_samples, self.batch_size, dtype=int
        )

        # Randomly pick the starts
        self._random.shuffle(starts)

        # Produce a sequence for each start
        batches = [
            np.arange(start, start + self.batch_size, dtype=int) for start in starts
        ]
        indices = np.concatenate(batches)

        # Wrap indexes around if the offset is non-zero
        indices = indices % self.num_samples

        return iter(indices)


class BootstrapBatchSampler(AbstractBatchSampler):
    """Sampler used for bootstrap sampling of the entire dataset.

    This is particularly useful for training an ensemble of networks
    (bagging).
    """

    name = "bootstrap"

    def __iter__(self) -> Iterator[int]:
        """Iterates over bootstrapped batches of data randomly picked
        from the dataset.
        """
        max_id = self.num_samples + 1 - self.batch_size
        starts = np.arange(0, max_id, self.batch_size, dtype=int)
        bootstrap_indices = np.random.choice(
            np.arange(self.num_samples), self.num_samples
        )
        batches = [
            bootstrap_indices[np.arange(start, start + self.batch_size, dtype=int)]
            for start in starts
        ]

        return iter(np.concatenate(batches))


class _Sized:
    """Minimal sized proxy used to drive paired secondary samplers."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class AbstractJointBatchSampler(Sampler):
    """Pair primary sampler indices with independently sampled secondary indices."""

    joint = True
    sampler_cls: type[AbstractBatchSampler]

    def __init__(
        self,
        dataset: Sized,
        batch_size: int,
        seed: int | None = None,
        drop_last: bool = True,
        pair_probability: float = 1.0,
    ) -> None:
        """Build matched primary and secondary index streams.

        Parameters
        ----------
        dataset : JointDataset
            Dataset with a ``secondary`` source to sample from.
        batch_size : int
            Number of primary samples to load per iteration.
        seed : int, optional
            Seed forwarded to the primary sampler. The secondary sampler uses
            the next seed so random policies are independent.
        drop_last : bool, default True
            Whether the primary sampler should drop incomplete batches.
        pair_probability : float, default 1.0
            Probability of pairing a primary index with a secondary index.
            When the draw fails, the sampler yields ``None`` for the secondary
            index and the joint dataset returns the primary sample unchanged.
        """
        super().__init__(data_source=None)

        if not getattr(dataset, "joint", False) or not hasattr(dataset, "secondary"):
            raise ValueError("Joint samplers require a JointDataset.")
        if pair_probability < 0.0 or pair_probability > 1.0:
            raise ValueError("`pair_probability` must be between 0 and 1.")

        self.primary_sampler = self.sampler_cls(dataset, batch_size, seed, drop_last)
        secondary_seed = None if seed is None else seed + 1
        self.secondary_sampler = self.sampler_cls(
            _Sized(len(self.primary_sampler)),
            batch_size,
            secondary_seed,
            drop_last,
        )
        self.secondary_size = len(dataset.secondary)
        self.batch_size = self.primary_sampler.batch_size
        self.drop_last = self.primary_sampler.drop_last
        self.num_samples = len(self.primary_sampler)
        self.pair_probability = pair_probability
        rng_seed = int(time.time()) if seed is None else seed + 2
        self._random = np.random.RandomState(seed=rng_seed)  # pylint: disable=E1101

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[int, int | None]]:
        primary_indices = list(self.primary_sampler)
        pair_mask = self._random.random_sample(len(primary_indices))
        pair_mask = pair_mask < self.pair_probability

        num_pairs = int(np.count_nonzero(pair_mask))
        paired_indices = np.asarray(list(self.secondary_sampler), dtype=int)
        paired_indices = paired_indices[:num_pairs] % self.secondary_size

        secondary_indices: list[int | None] = []
        paired_pos = 0
        for paired in pair_mask:
            if paired:
                secondary_indices.append(int(paired_indices[paired_pos]))
                paired_pos += 1
            else:
                secondary_indices.append(None)

        return iter(zip(primary_indices, secondary_indices))


class JointSequentialBatchSampler(AbstractJointBatchSampler):
    """Sequential primary sampling with paired sequential secondary sampling."""

    name = "joint_sequential"
    sampler_cls = SequentialBatchSampler


class JointRandomSequenceBatchSampler(AbstractJointBatchSampler):
    """Random-sequence primary sampling with paired random secondary sampling."""

    name = "joint_random_sequence"
    sampler_cls = RandomSequenceBatchSampler


class JointBootstrapBatchSampler(AbstractJointBatchSampler):
    """Bootstrap primary sampling with paired bootstrap secondary sampling."""

    name = "joint_bootstrap"
    sampler_cls = BootstrapBatchSampler


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    """

    def __init__(
        self, sampler: AbstractBatchSampler, num_replicas: int, rank: int
    ) -> None:
        """Convert a basic sampler to an instance of a distributed sampler.

        Parameters
        ----------
        sampler : Sampler
            Input torch sampler
        num_replicas : int
            Number of distributed samplers running concurrently
        rank : int
            Rank of the current sampler

        Notes
        -----
        Input sampler is assumed to be of constant size.
        """
        # Make sure the batch_size is a multiple of the number of replicas
        assert sampler.batch_size % num_replicas == 0, (
            f"The `batch_size` ({sampler.batch_size}) must be a multiple "
            f"of the number of replicas ({num_replicas}) in the "
            "distributed training process."
        )

        # Initialiaze the parent distributed sampler
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=sampler.drop_last,
            shuffle=False,
        )

        # Store the underlying sampler and its parameters
        self.sampler = sampler
        self.batch_size = sampler.batch_size

    def __iter__(self) -> Iterator[int]:
        """Overrides the basic iterator with one that takes into account
        the number of replicas and the rank of the sampler.
        """
        # Fetch the list of non-distributed indices
        indices = list(self.sampler)

        # If the number of entries is not a multiple of the number of replicas,
        # must pad the end.
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]

        assert len(indices) == self.total_size

        # Subsample by keeping the indices sequential between each minibatch.
        # This is crucial to preserve contiguousness in sequential samplers.
        minibatch_size = self.batch_size / self.num_replicas
        ranks = (np.arange(self.total_size) // minibatch_size) % self.num_replicas

        indices = [indices[i] for i in np.where(ranks == self.rank)[0]]
        assert len(indices) == self.num_samples

        return iter(indices)
