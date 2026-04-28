Input/Output Module
===================

The ``spine.io`` module handles data ingress and egress for SPINE jobs. It includes framework-independent readers and writers as well as PyTorch-aware dataset, collation, augmentation, and sampling utilities used during model training and inference.

.. currentmodule:: spine.io

.. automodule:: spine.io
   :no-members:

Overview
--------

The I/O layer is split into two main parts:

- **Core I/O** for framework-independent reading and writing of HDF5, LArCV, ROOT, and related event data
- **Torch I/O** for datasets, collation, augmentation, overlays, and sampling in ML workflows

This is the first stage of the driver pipeline and the point where external detector data is mapped into SPINE's internal data structures.

File Readers
------------

.. autosummary::
   :toctree: generated

   core.read.HDF5Reader
   core.read.LArCVReader

File Writers
------------

.. autosummary::
   :toctree: generated

   core.write.HDF5Writer
   core.write.CSVWriter

Data Processing
---------------

Tools for dataset preparation, augmentation, collation, and sampling.

.. autosummary::
   :toctree: generated

   torch.collate
   torch.dataset
   torch.sample
   torch.augment
   torch.overlay
   torch.factories
