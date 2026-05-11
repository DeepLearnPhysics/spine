Input/Output Module
===================

The ``spine.io`` module handles data ingress and egress for SPINE jobs. It
provides readers and writers for event data, parsers that translate raw
storage products into SPINE parser objects, and the dataset/collation tools
used during model training and inference.

.. currentmodule:: spine.io

.. automodule:: spine.io
   :no-members:

Overview
--------

The I/O layer is organized into a few cooperating pieces:

- **Readers** expose event products from on-disk formats such as HDF5 and
  LArCV.
- **Writers** persist flat outputs and staged cache products.
- **Parsers** convert raw reader outputs into SPINE parser products used by
  downstream code.
- **Datasets and pipeline utilities** bridge readers/parsers into PyTorch
  data loading workflows.

This is the first stage of the driver pipeline and the point where external
detector data is mapped into SPINE's internal data structures.

File Readers
------------

.. autosummary::
   :toctree: generated

   read.HDF5Reader
   read.LArCVReader
   read.StageHDF5Reader

File Writers
------------

.. autosummary::
   :toctree: generated

   write.HDF5Writer
   write.CSVWriter
   write.StageHDF5Writer

Datasets
--------

The dataset layer bridges low-level readers and parser logic into PyTorch
``Dataset`` objects. The staged cache workflow is exposed through the HDF5
dataset and the mixed LArCV/HDF5 dataset.

.. autosummary::
   :toctree: generated

   dataset.LArCVDataset
   dataset.HDF5Dataset
   dataset.MixedDataset

Parsers
-------

Parsers translate raw reader outputs into framework-neutral parser products.
The HDF5 parser layer includes generic tensor, index, and object parsers for
cached data products.

.. autosummary::
   :toctree: generated

   parse.base
   parse.data
   parse.clean_data
   parse.hdf5.tensor
   parse.hdf5.index
   parse.hdf5.object
   parse.larcv.misc
   parse.larcv.sparse
   parse.larcv.cluster
   parse.larcv.particle

Data Pipeline Utilities
-----------------------

Tools for dataset preparation, augmentation, collation, and sampling.

.. autosummary::
   :toctree: generated

   collate
   sample
   augment
   overlay
   factories
