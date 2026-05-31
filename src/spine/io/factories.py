"""Functions that instantiate IO tools from configuration blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from warnings import warn

from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.factory import instantiate, module_dict

from . import read, write

READER_DICT = module_dict(read)
WRITER_DICT = module_dict(write)

__all__ = [
    "reader_factory",
    "writer_factory",
    "loader_factory",
    "dataset_factory",
    "sampler_factory",
    "collate_factory",
]


def reader_factory(reader_cfg: Mapping[str, Any] | str) -> Any:
    """Instantiate a reader from a configuration block.

    The configured ``name`` must match a reader class exported from
    :mod:`spine.io.read`.

    Parameters
    ----------
    reader_cfg : Mapping[str, Any] or str
        Reader configuration mapping or the short reader name.

    Returns
    -------
    object
        Instantiated reader object.
    """
    # Initialize reader
    return instantiate(READER_DICT, reader_cfg)


def writer_factory(
    writer_cfg: Mapping[str, Any] | str,
    prefix: str | list[str] | None = None,
    split: bool = False,
) -> Any:
    """Instantiate a writer from a configuration block.

    The configured ``name`` must match a writer class exported from
    :mod:`spine.io.write`.

    Parameters
    ----------
    writer_cfg : Mapping[str, Any] or str
        Writer configuration mapping or the short writer name.
    prefix : str or list[str], optional
        Input file prefix or per-file list of prefixes used to derive output
        names when the writer supports prefix-based naming.
    split : bool, default False
        Request one output file per input file. Writers that do not support
        unsplit output may reject ``split=False`` explicitly.

    Returns
    -------
    object
        Instantiated writer object.
    """
    # Initialize writer
    extra_kwargs = {}
    if prefix is not None:
        extra_kwargs["prefix"] = prefix
    if split:
        extra_kwargs["split"] = split

    return instantiate(WRITER_DICT, writer_cfg, **extra_kwargs)


def loader_factory(
    dataset: Mapping[str, Any] | str,
    dtype: str,
    batch_size: int | None = None,
    minibatch_size: int | None = None,
    shuffle: bool = True,
    sampler: Mapping[str, Any] | str | None = None,
    num_workers: int = 0,
    collate_fn: Mapping[str, Any] | str | None = None,
    entry_list: list[int] | None = None,
    distributed: bool = False,
    world_size: int = 0,
    rank: int | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a PyTorch ``DataLoader`` from configuration.

    Parameters
    ----------
    dataset : mapping or str
        Dataset configuration mapping or short dataset name.
    dtype : str
        Floating-point dtype passed to the dataset factory.
    batch_size : int, optional
        Global batch size. Mutually exclusive with ``minibatch_size``.
    minibatch_size : int, optional
        Per-process batch size. Mutually exclusive with ``batch_size``.
    shuffle : bool, default True
        Whether to shuffle batches in the underlying loader.
    sampler : mapping or str, optional
        Sampler configuration mapping or short sampler name.
    num_workers : int, default 0
        Number of loader worker processes.
    collate_fn : mapping or str, optional
        Collate function configuration mapping or short collate name.
    entry_list : list[int], optional
        Explicit subset of dataset entries to expose.
    distributed : bool, default False
        If ``True``, wrap the sampler for distributed loading.
    world_size : int, default 0
        Number of distributed processes/devices.
    rank : int, optional
        Distributed process rank. Required when ``distributed=True``.
    **kwargs : dict
        Extra keyword arguments forwarded to ``torch.utils.data.DataLoader``.

    Returns
    -------
    torch.utils.data.DataLoader
        Instantiated data loader.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to use loader_factory.")

    from torch.utils.data import DataLoader

    # Process the batch size, make sure it is sensible
    if batch_size is not None and minibatch_size is not None:
        raise ValueError("Provide either `batch_size` or `minibatch_size`, not both.")

    if batch_size is not None:
        if world_size != 0 and (batch_size % world_size) != 0:
            raise ValueError("The batch_size must be a multiple of the number of GPUs.")
        minibatch_size = batch_size // max(world_size, 1)
    elif minibatch_size is not None:
        batch_size = minibatch_size * max(world_size, 1)
    else:
        raise ValueError("Provide either `batch_size` or `minibatch_size`, not both.")

    # Initialize the dataset
    torch_dataset = dataset_factory(dataset, entry_list, dtype)

    # Initialize the sampler
    if sampler is None and getattr(torch_dataset, "joint", False):
        raise ValueError("JointDataset requires an explicit joint sampler.")
    if sampler is not None:
        sampler = sampler_factory(
            sampler, torch_dataset, batch_size, distributed, world_size, rank
        )

    # Initialize the collate function
    if collate_fn is not None:
        collate_fn = collate_factory(
            collate_fn, torch_dataset.data_types, torch_dataset.overlay_methods
        )

    # Initialize the loader
    return DataLoader(
        torch_dataset,
        batch_size=minibatch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def dataset_factory(
    dataset_cfg: Mapping[str, Any] | str,
    entry_list: list[int] | None = None,
    dtype: str | None = None,
) -> Any:
    """Instantiate a dataset from configuration.

    Parameters
    ----------
    dataset_cfg : Mapping[str, Any] or str
        Dataset configuration mapping or short dataset name.
    entry_list : list[int], optional
        Explicit subset of dataset entries to expose. When provided here, it
        overrides any ``entry_list`` already present in ``dataset_cfg``.
    dtype : str, optional
        Floating-point dtype forwarded to the dataset constructor.

    Returns
    -------
    object
        Instantiated dataset object.
    """
    from . import dataset

    # Get the dataset class dictionary
    dataset_dict = module_dict(dataset)

    # Append the entry_list if it is provided independently
    if entry_list is not None:
        dataset_name = (
            dataset_cfg if isinstance(dataset_cfg, str) else dataset_cfg.get("name")
        )
        if dataset_name in ("joint", "JointDataset"):
            raise ValueError(
                "`entry_list` must be configured inside `base`, `primary`, "
                "or `secondary` for JointDataset."
            )
        warn(
            "You are manually overwriting the existing `entry_list` "
            "argument provided in the configuration file."
        )
        dataset_cfg = (
            {"name": dataset_cfg} if isinstance(dataset_cfg, str) else dict(dataset_cfg)
        )
        dataset_cfg["entry_list"] = entry_list

    # Initialize dataset
    extra_kwargs: dict[str, Any] = {"dtype": dtype}

    return instantiate(dataset_dict, dataset_cfg, **extra_kwargs)


def sampler_factory(
    sampler_cfg: Mapping[str, Any] | str,
    dataset: Any,
    minibatch_size: int,
    distributed: bool = False,
    num_replicas: int = 1,
    rank: int | None = None,
) -> Any:
    """Instantiate a sampler from configuration.

    Parameters
    ----------
    sampler_cfg : mapping or str
        Sampler configuration mapping or short sampler name.
    dataset : object
        Dataset instance used to initialize the sampler.
    minibatch_size : int
        Per-process batch size passed to the sampler.
    distributed : bool, default False
        If ``True``, wrap the sampler in ``DistributedProxySampler``.
    num_replicas : int, default 1
        Number of distributed processes/devices.
    rank : int, optional
        Distributed process rank. Required when ``distributed=True``.

    Returns
    -------
    object
        Instantiated sampler object, optionally wrapped for distributed
        loading.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to use sampler_factory.")

    if distributed and rank is None:
        raise ValueError("A distributed sampler requires an explicit integer `rank`.")

    from . import sample

    # Get the sampler class dictionary
    sampler_dict = module_dict(sample)

    # Initialize sampler
    sampler_obj = instantiate(
        sampler_dict, sampler_cfg, dataset=dataset, batch_size=minibatch_size
    )

    # Joint datasets consume tuple indexes; standard datasets consume scalars.
    is_joint_dataset = getattr(dataset, "joint", False)
    is_joint_sampler = getattr(sampler_obj, "joint", False)
    if is_joint_dataset != is_joint_sampler:
        expected = "joint" if is_joint_dataset else "standard"
        got = "joint" if is_joint_sampler else "standard"
        raise ValueError(
            f"Cannot use a {got} sampler with a {expected} dataset. "
            "Use a joint sampler with JointDataset and a standard sampler "
            "with standard datasets."
        )

    # If we are working a distributed environment, wrap the sampler
    if distributed:
        sampler_obj = sample.DistributedProxySampler(sampler_obj, num_replicas, rank)

    return sampler_obj


def collate_factory(
    collate_cfg: Mapping[str, Any] | str,
    data_types: Mapping[str, str],
    overlay_methods: Mapping[str, str],
) -> Any:
    """Instantiate a collate function from configuration.

    Parameters
    ----------
    collate_cfg : Mapping[str, Any] or str
        Collate configuration mapping or short collate function name.
    data_types : Mapping[str, str]
        Mapping from parser output keys to their declared data type.
    overlay_methods : Mapping[str, str]
        Mapping from parser output keys to the overlay method used when
        combining data from multiple sources.

    Returns
    -------
    collections.abc.Callable
        Instantiated collate callable.
    """
    from . import collate

    # Get the collate function class dictionary
    collate_dict = module_dict(collate)

    return instantiate(
        collate_dict,
        collate_cfg,
        "collate_fn",
        data_types=data_types,
        overlay_methods=overlay_methods,
    )
