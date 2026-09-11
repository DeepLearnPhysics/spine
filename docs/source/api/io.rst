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
- **Writers** persist flat outputs and transactional cache stages.
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
   read.CacheReader

Entry Filtering
---------------

File-aware filtering is performed before ordinary reader selection. The
generic scan and manifest APIs delegate physical inspection to a backend such
as :class:`filter.LArCVEntryInspector`.

.. autosummary::
   :toctree: generated

   filter.EntryInspector
   filter.LArCVEntryInspector
   filter.scan_sources
   filter.build_manifest
   filter.load_entry_filter
   filter.eligible_entries_from_manifest
   filter.eligible_cache_entries_from_manifest

File Writers
------------

.. autosummary::
   :toctree: generated

   write.HDF5Writer
   write.CacheWriter

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
``Dataset`` objects. SPINE cache repositories have their own dataset type and
can also be paired with raw input through the mixed dataset.

.. autosummary::
   :toctree: generated

   dataset.LArCVDataset
   dataset.CacheDataset
   dataset.HDF5Dataset
   dataset.MixedDataset
   dataset.JointDataset

Sharded cache repositories
--------------------------

SPINE's production cache is a directory, conventionally named
``train.spine-cache``, containing an atomic ``manifest.json`` and immutable
HDF5 V2 shards. Each source and processing-stage generation owns one shard.
Publishing a later stage writes only that stage and atomically replaces the
small manifest; it never recopies earlier cache products. Readers snapshot the
manifest at initialization. Adding new downstream stages does not disturb an
active reader, but replacing stages while jobs are reading the same repository
is intentionally unsupported.

Create the first stage with an explicit repository path:

.. code-block:: yaml

   base:
     unwrap: true

   io:
     writer:
       name: cache
       path: /path/to/train.spine-cache
       stage: deghosting
       keys: [data_adapt, seg_pred, orig_index]

A later cache-backed job can omit the writer path. The I/O manager discovers
the input repository and publishes the new stage back to it:

.. code-block:: yaml

   base:
     unwrap: true

   io:
     loader:
       minibatch_size: 64
       num_workers: 4
       dataset:
         name: cache
         path: /path/to/train.spine-cache
         stage: deghosting
     writer:
       name: cache
       stage: fragmentation

Mixed raw/cache training remains a normal mixed dataset. The child roles are
explicit: ``primary`` selects the authoritative dataset implementation, while
``cache`` is a manifest-backed SPINE cache. Storage-format aliases such as
``larcv`` and ``hdf5`` are intentionally not part of the mixed-dataset
contract:

.. code-block:: yaml

   dataset:
     name: mixed
     primary:
       name: larcv
       file_keys: /path/to/raw/*.root
       schema: {...}
     cache:
       path: /path/to/train.spine-cache
       stage: deghosting

When a scheduler-array task opens only a subset of the primary source files,
the mixed dataset automatically projects the complete repository onto matching
source identities in primary-file order. Cache shards belonging to other tasks
are not opened and do not participate in cardinality checks.

Set ``overwrite_stage: true`` to publish a replacement generation. The new
manifest is committed first; SPINE then removes the replaced generation and
every transitive descendant derived from it. A failed write or publication
therefore leaves the previously published cache untouched, while a successful
pipeline does not accumulate obsolete cache payloads. The manifest records the
exact upstream stage generations consumed by each new stage.

Hard process termination may leave unpublished transaction directories or
other unreachable generations. They can be inspected and removed explicitly::

   spine-cache gc /path/to/train.spine-cache --dry-run
   spine-cache gc /path/to/train.spine-cache

Garbage collection never removes files referenced by the current manifest and
defaults to a one-day minimum inactivity age. Active writers refresh a pending
transaction heartbeat, protecting both their pending data and shards being
moved into publication. This command is recovery maintenance, not a required
step in a successful production pipeline.

Scheduler arrays may build one repository without a final HDF5 merge by
setting ``parallel: true`` on every cache writer. Each task publishes a
disjoint set of source shards under the manifest lock. The first stage grows
the source roster as contributions arrive. For subsequent stages, the
manifest records the stage as incomplete—and readers reject it—until shards
for the complete established source roster have arrived::

   io:
     writer:
       name: cache
       path: /path/to/train.spine-cache
       stage: deghosting
       parallel: true
       expected_sources: 400

``expected_sources`` is the total number of source files with accepted entries
across all array tasks. A source rejected completely by an entry filter does
not need an empty shard and is not included in this count. The field provides
the completion barrier without a merge job and ensures a missing task cannot
leave a partial first stage looking valid. Parallel contributions must expose
identical product schemas and lineage. During initial construction, their
source sets must be disjoint.

Parallel replacement combines ``parallel: true`` with
``overwrite_stage: true``. Contributions are recorded as a hidden replacement
while readers continue seeing the old complete stage. Repeating a source
replaces that source's hidden contribution, which makes individual array tasks
safe to retry. When the final expected source arrives, the manifest atomically
activates the assembled replacement and invalidates the old stage plus all of
its descendants. No merge or finalization job is required.

The ordinary CLI path options understand this cache contract. For example,
``--output train.spine-cache`` overrides a cache writer's ``path``, while
``--source train.spine-cache`` and ``--val-source train-val.spine-cache``
override a cache dataset's ``path``. A cache repository is a single logical
input, so ``--source-list`` and multiple direct paths are rejected. Canonical
mixed datasets use role-qualified values such as
``--source primary=raw.root cache=train.spine-cache``; the same contract
applies to ``--val-source``.

Data augmentation
-----------------

These classes are selected by ``name`` in a dataset's ``augment``
configuration.  Their class pages list the available transformation options.

.. autosummary::
   :toctree: generated

   augment.AugmentManager
   augment.CalibrationAugment
   augment.CropAugment
   augment.FlipAugment
   augment.JitterAugment
   augment.MaskAugment
   augment.ResponseAugment
   augment.RotateAugment
   augment.TranslateAugment

Calibration-aware response variation keeps the nominal detector description
separate from the distributions sampled during training.  For example, this
configuration maps nominal raw ADC response into a response with independently
varied lifetime and gain, then adds image-level noise::

   augment:
     calibration:
       features:
         data: 0
       sources: sources
       run_info: run_info
       nominal:
         gain:
           gain: 185.0
         lifetime:
           lifetime: 3000.0
           driftv: 0.16
       throws:
         gain:
           gain:
             distribution: normal
             sigma: 0.05
             relative: true
             scope: tpc
             clip: [0.8, 1.2]
         lifetime:
           lifetime:
             distribution: normal
             sigma: 0.10
             relative: true
             scope: image
             clip: [0.5, 1.5]
       noise:
         scale: 0.5
         scope: voxel

The default ``vary_response`` mode applies the nominal correction and then the
inverse thrown calibration.  ``calibrate`` and ``simulate`` are available for
inputs that should cross the calibration boundary directly.  Arbitrary signal
response functions require an explicit ``inverse_response_func`` whenever an
inverse mode is used; stochastic noise is applied only after deterministic
response transformations.

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
