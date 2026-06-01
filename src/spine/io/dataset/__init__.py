"""Torch-backed dataset adapters for SPINE IO.

The dataset layer sits between low-level readers and PyTorch DataLoaders. It is
responsible for:

- exposing ``__len__`` and ``__getitem__`` for torch
- converting raw reader outputs into parser products
- attaching augmentation, collate-type, and overlay metadata
- composing source datasets when training needs aligned cache products
  (``MixedDataset``) or unaligned overlay pairs (``JointDataset``)
"""

from .hdf5 import HDF5Dataset
from .joint import JointDataset
from .larcv import LArCVDataset
from .mixed import MixedDataset

__all__ = ["HDF5Dataset", "JointDataset", "LArCVDataset", "MixedDataset"]
