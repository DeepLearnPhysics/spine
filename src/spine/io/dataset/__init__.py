"""Torch-backed dataset adapters for SPINE IO.

The dataset layer sits between low-level readers and the DataLoader. It is
responsible for:

- exposing ``__len__`` and ``__getitem__`` for torch
- converting raw reader outputs into parser products
- attaching augmentation, collate-type, and overlay metadata
"""

from .hdf5 import HDF5Dataset
from .larcv import LArCVDataset
from .mixed import MixedDataset

__all__ = ["HDF5Dataset", "LArCVDataset", "MixedDataset"]
