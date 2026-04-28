Input/Output Module
===================

The ``spine.io`` module provides comprehensive tools for data input/output operations with support for multiple file formats.

Overview
--------

The I/O module is organized into:

- **Core I/O**: Framework-independent file reading/writing (HDF5, LArCV, ROOT)
- **PyTorch I/O**: Dataset loaders and batch processing for ML training
- **Parsers**: Extract and transform data from various formats
- **Collaters**: Batch multiple samples for efficient training

File Readers
------------

Reader classes for various file formats. All readers inherit from ``ReaderBase``
and share common attributes for file management, indexing, and metadata.

HDF5 Reader
~~~~~~~~~~~

.. autoclass:: spine.io.core.read.HDF5Reader
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__

LArCV Reader
~~~~~~~~~~~~

.. autoclass:: spine.io.core.read.LArCVReader
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__

File Writers
------------

Writer classes for output operations.

HDF5 Writer
~~~~~~~~~~~

.. autoclass:: spine.io.core.write.HDF5Writer
   :members:
   :show-inheritance:

CSV Writer
~~~~~~~~~~

.. autoclass:: spine.io.core.write.CSVWriter
   :members:
   :show-inheritance:

Data Processing
---------------

Tools for data augmentation, collation, and sampling.

Collaters
~~~~~~~~~

.. automodule:: spine.io.collate
   :members:
   :show-inheritance:

Samplers
~~~~~~~~

.. automodule:: spine.io.sample
   :members:
   :show-inheritance:

Augmentation
~~~~~~~~~~~~

.. automodule:: spine.io.augment
   :members:
   :show-inheritance:

Parsers
-------

Data extraction and transformation utilities.

.. automodule:: spine.io.parse
   :members:
   :show-inheritance:
