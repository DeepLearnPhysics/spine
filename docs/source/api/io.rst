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

Manager
-------

.. autosummary::
   :toctree: generated

   manager.IOManager

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
   write.StageHDF5Writer

Tabular metric logs are written by :class:`spine.logging.CSVLogger`, not by
the generic event-output writer interface.

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

For high-level workflows which need only scalar and fixed-width object
attributes, the V2 reader can skip all variable-value pools:

.. code-block:: yaml

   dataset:
     name: hdf5
     fixed_only: true

Full loading remains the default. ``fixed_only`` is intentionally restricted
to format version 2 files and omits variable attributes such as indexes,
matches, strings, and variable-width vectors. When classes are rebuilt, those
attributes retain their class defaults, so derived properties which depend on
them must not be used. Set ``build_classes: false`` to retain the stored
derived fields directly in the returned object dictionaries.

Analysis-only workflows may instead request projected multi-event chunks:

.. code-block:: yaml

   io:
     reader:
       name: hdf5
       file_keys: output.h5
       columnar: true
       chunk_size: 1024

Columnar mode is a reader-wide policy. The analysis manager supplies the union
of fields requested by its scripts, and the reader returns flattened object
columns with an ``event_offsets`` boundary vector for each product. Version 2
uses its native offsets and fixed compound rows; version 1 projects the legacy
compound dataset through event region references. Legacy files must already
contain every requested scalar field, such as ``best_match_id``.

The driver currently restricts columnar mode to analysis-only configurations:
all configured scripts must implement ``process_columnar``, and model,
construction, post-processing, and ordinary output-writer blocks are rejected.

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
   dataset.JointDataset

Extending staged caches
-----------------------

When a staged HDF5 cache is both the input and output of a driver job, SPINE
automatically writes the new stage to a temporary sidecar file. The canonical
cache remains read-only while the loader is active, so HDF5 dataset reads may
use multiple workers without competing with a writer handle. After successful
processing, finalization builds a merged temporary copy beside each canonical
cache and publishes it with an atomic file replacement. A failed run removes
its uncommitted sidecars and leaves the canonical cache unchanged.

No additional writer option is required for the usual same-file workflow. It
is selected when both reader and writer use ``stage_hdf5`` and the writer has
no explicit ``file_name`` or ``directory``. For example:

.. code-block:: yaml

   base:
     split_output: true

   io:
     loader:
       minibatch_size: 64
       num_workers: 4
       shuffle: false
       dataset:
         name: hdf5
         staged: true
         stage: fragmentation
         file_keys: null
     writer:
       name: stage_hdf5
       file_name: null
       stage: particle_aggregation
       overwrite_stage: true

An explicit output destination retains the ordinary separate-output behavior.
``sidecar: false`` may be used to opt out of automatic same-file sidecars, but
direct writes again require the caller to avoid concurrent handles. Each
canonical file is replaced atomically; a multi-file job is validated in full
before publication, but is not a single filesystem-wide transaction. During
finalization, the destination filesystem must have enough free space for one
temporary copy of the canonical cache plus the new stage sidecar.

Data augmentation
-----------------

These classes are selected by ``name`` in a dataset's ``augment``
configuration.  Their class pages list the available transformation options.

.. autosummary::
   :toctree: generated

   augment.AugmentManager
   augment.CropAugment
   augment.FlipAugment
   augment.JitterAugment
   augment.MaskAugment
   augment.ResponseAugment
   augment.RotateAugment
   augment.TranslateAugment

Parsers
-------

Parsers translate raw reader outputs into framework-neutral parser products.
The HDF5 parser layer includes generic tensor, index, and object parsers for
cached data products.

The parser classes below are the values accepted by dataset ``schema``
entries.  Their constructor signatures are the schema's configurable
parameters.

.. autosummary::
   :toctree: generated

   parse.hdf5.tensor.HDF5TensorParser
   parse.hdf5.tensor.HDF5ClusterTensorParser
   parse.hdf5.tensor.HDF5FeatureTensorParser
   parse.hdf5.index.HDF5IndexParser
   parse.hdf5.index.HDF5IndexListParser
   parse.hdf5.index.HDF5EdgeIndexParser
   parse.hdf5.object.HDF5ObjectParser
   parse.hdf5.object.HDF5ObjectListParser
   parse.hdf5.cluster.HDF5ClusterLabelParser
   parse.larcv.misc.LArCVMetaParser
   parse.larcv.misc.LArCVRunInfoParser
   parse.larcv.misc.LArCVFlashParser
   parse.larcv.misc.LArCVCRTHitParser
   parse.larcv.misc.LArCVTriggerParser
   parse.larcv.sparse.LArCVSparse2DParser
   parse.larcv.sparse.LArCVSparse3DParser
   parse.larcv.sparse.LArCVSparse3DAggregateParser
   parse.larcv.sparse.LArCVSparse3DChargeRescaledParser
   parse.larcv.sparse.LArCVSparse3DGhostParser
   parse.larcv.cluster.LArCVCluster2DParser
   parse.larcv.cluster.LArCVCluster3DParser
   parse.larcv.cluster.LArCVCluster3DAggregateParser
   parse.larcv.cluster.LArCVCluster3DChargeRescaledParser
   parse.larcv.particle.LArCVParticleParser
   parse.larcv.particle.LArCVNeutrinoParser
   parse.larcv.particle.LArCVParticlePointParser
   parse.larcv.particle.LArCVParticleCoordinateParser
   parse.larcv.particle.LArCVVertexPointParser
   parse.larcv.particle.LArCVParticleGraphParser
   parse.larcv.particle.LArCVSingleParticlePIDParser
   parse.larcv.particle.LArCVSingleParticleEnergyParser

Parser implementation modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   parse.base
   parse.clean_data
   parse.hdf5.tensor
   parse.hdf5.index
   parse.hdf5.object
   parse.hdf5.cluster
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
   unwrap
   factories

Batch samplers
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   sample.SequentialBatchSampler
   sample.RandomSequenceBatchSampler
   sample.BootstrapBatchSampler
   sample.JointSequentialBatchSampler
   sample.JointRandomSequenceBatchSampler
   sample.JointBootstrapBatchSampler
   sample.DistributedProxySampler
   collate.CollateAll
