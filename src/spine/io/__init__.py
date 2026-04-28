"""I/O tools for reading and writing neutrino physics data.

This module provides comprehensive data input/output capabilities for various file formats
commonly used in neutrino physics experiments, with support for both PyTorch-dependent
and independent operations.

Core I/O functionality:

- ``core`` for framework-independent readers and writers.
- ``torch`` for PyTorch datasets, collation, sampling, and augmentation.

Supported formats include LArCV, HDF5, ROOT, and NumPy-friendly tensor data.

Example HDF5 configuration
--------------------------

.. code-block:: yaml

   io:
     writer:
       name: HDF5Writer
       file_name: output.h5
       input_keys: [input_data, truth_labels]
       result_keys: [predictions, metrics]

PyTorch-dependent functionality is conditionally imported and available
only when PyTorch is installed. Check ``TORCH_IO_AVAILABLE``.
"""

# Always import PyTorch-independent core functionality
from .core import *

# Conditionally import PyTorch-dependent functionality
try:
    from .torch import *

    TORCH_IO_AVAILABLE = True
except ImportError:
    TORCH_IO_AVAILABLE = False
