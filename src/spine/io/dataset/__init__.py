"""Framework-neutral event dataset adapters for SPINE IO.

The dataset layer sits between low-level readers and optional data loaders. It is
responsible for:

- exposing map-style ``__len__`` and ``__getitem__`` access
- converting raw reader outputs into parser products
- attaching augmentation, collate-type, and overlay metadata
- composing source datasets when training needs aligned cache products
  (``MixedDataset``) or unaligned overlay pairs (``JointDataset``)
"""

from .cache import CacheDataset
from .hdf5 import HDF5Dataset
from .joint import JointDataset
from .larcv import LArCVDataset
from .mixed import MixedDataset

__all__ = [
    "CacheDataset",
    "HDF5Dataset",
    "JointDataset",
    "LArCVDataset",
    "MixedDataset",
]
