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

HDF5 format versions
--------------------

Flat SPINE HDF5 files are self-describing. The ``/info`` attributes separate
the producing software release from the physical file layout:

- ``spine_version`` identifies the SPINE release which produced the file.
- ``format`` is ``spine_hdf5`` for flat event files.
- ``format_version`` is ``1`` for the legacy region-reference/VLEN layout or
  ``2`` for the offset-based layout.

Files written before explicit layout versioning have no ``format_version`` and
are treated as version 1. :class:`read.HDF5Reader` detects both layouts
automatically. Select version 2 for new output explicitly during its rollout:

.. code-block:: yaml

   writer:
     name: hdf5
     format_version: 2

Version 2 keeps derived scalar and fixed-width properties directly available
in each product's ``fixed`` compound dataset. Variable-length properties use
dtype-specific pools under ``variables``. Each pool declares its ordered field
names in the ``fields`` attribute and has one flat ``values`` dataset. The
corresponding integer offset row is stored directly in the object's ``fixed``
record. Product ``event_offsets`` map event ``i`` to rows
``event_offsets[i]:event_offsets[i + 1]`` without HDF5 region references.
Appending data with a different format version is rejected.

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
