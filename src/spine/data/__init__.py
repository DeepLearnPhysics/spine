"""Data structures and containers.

This module defines all core data structures used throughout the SPINE package,
from low-level detector data to high-level physics objects.

Core namespaces:

- ``larcv`` for detector-level and generator-level input objects.
- ``out`` for high-level reconstructed and truth objects.
- ``batch`` for batched tensor, edge-index, and index containers.
- ``list`` for generic list-backed containers.

Key features:

- Shared base classes with merged docstrings and validation helpers.
- Units metadata through the ``field_units`` interface.
- Serialization support for I/O and analysis workflows.
- Conditional batch helpers when PyTorch is available.

Example
-------

.. code-block:: python

  from spine.data import Particle, TensorBatch

  particle = Particle(id=0, pid=13, momentum=[1.2, 0.5, 2.1])
  batch = TensorBatch(data_list, batch_size=32)
"""

from .batch import *
from .larcv import *
from .list import *
from .out import *
