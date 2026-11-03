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
- **Datasets** combine readers and parsers into framework-neutral event
  products. They may be traversed directly or passed to a PyTorch loader.
- **Pipeline utilities** add batching, collation, sampling, and unwrapping for
  model workflows.

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
   read.SuperaReader
   read.LArCVReader
   read.CacheReader

Native pysupera input
---------------------

Use ``provider: supera`` for pysupera **4.1.0** files. This is a separate
format from SPINE HDF5 V1/V2; ``provider: hdf5`` does not read these files.
Install ``spine[supera]`` for the HDF5 compression filters used by upstream
productions. No pysupera or LArCV runtime is required.

.. code-block:: yaml

   io:
     loader:
       minibatch_size: 1
       num_workers: 0
       dataset:
         provider: supera
         file_keys: supera_pixel_j000000-000004.h5
         schema:
           data: {provider: tensor}
           seg_label: {provider: segmentation}
           clust_label: {provider: cluster_label}
           ppn_label: {provider: particle_points}
           coord_label: {provider: particle_coords}
           graph_label: {provider: particle_graph}
           particles: {provider: particle}
           meta: {provider: meta}
           source_ids: {provider: source_ids}
           event_info: {provider: event_info}
       collate_fn: {provider: all}

``SuperaDataset`` reuses the HDF5 dataset's parsing, collation and augmentation
interface with a format-specific parser registry. ``SuperaReader`` supports
multiple files, ordinary entry selection, source-file provenance and projected
reads. Its default raw products are ``points``, ``particles``, ``inter``,
``events`` and ``metadata``. Particle and interaction point ranges are rebased
to the event's point array. Interaction particle ranges are likewise rebased.
The ``event_info`` parser preserves every stored event/input/job identity as a
named integer tensor, including ``event_id``, ``input``, ``input_event``,
``jaxtpc_entry``, ``job_event`` and ``job_id`` in the sample. It does not invent
run/subrun numbers. The raw identities remain available as ``events``.

Physical semantics
~~~~~~~~~~~~~~~~~~

The implementation was checked against `pysupera commit dab92d8
<https://github.com/drinkingkazu/pysupera/tree/dab92d8fc850c09d0ca85cff16c6f36c28cf6d4b>`_,
in particular ``_run.py``, ``layout.py``, ``io_v3.py``, ``provenance.py`` and
``torchdata.py``:

* With JAXTPC input, ``points/flat`` contains **truth energy deposits**, not
  measured hits. Coordinates are mm, ``t`` is microseconds, ``dE`` is MeV,
  ``dX`` is cm, angles are radians and ``p`` is MeV/c. ``hit_energy=charge``
  describes the hit-based partitioning input; it does not change ``dE`` into
  charge. Measured hit coordinates and charge require the separate JAXTPC file.
* ``groups`` is indexed by JAXTPC deposit group, concatenated across volumes.
  Its rows do not align with ``points/flat``. In the supplied 1,000-event
  production there are 54,863,490 groups and 31,871,012 point rows.
  ``volume_offsets`` identifies each volume's start in the event's group array.
* Fragment representatives satisfy ``frag_id == id``. Their half-open
  ``frag_pc_start:frag_pc_end`` ranges partition the point rows. Instance ranges
  also partition the points, with non-LE fragments preceding the LE portion
  starting at ``inst_pc_le_start``. ``frag_inst_id`` joins each fragment to
  its instance. The reader never assumes group/point alignment.
* The writer retains sparse particle IDs. Despite the upstream reader module's
  row-index wording, these must be resolved through an ID-to-row map. SPINE
  particle IDs, cluster-sidecar ``particle`` IDs, ancestor links, graph nodes
  and PPN references are all event-local particle-table rows. ``source_ids``
  preserves the sparse source ID for each row, without shifting it in a batch.
* Semantic codes map explicitly: pysupera shower/track/delta/Michel/LE
  ``1/2/3/4/5`` become SPINE ``0/1/3/2/4``. Non-LE points use the instance's
  semantic class; LE fragments retain LE even inside a track or shower.
  Unknown point semantics are rejected by the five-class training parsers.
* Particle ``start_*``/``end_*`` describe Geant4 creation/destruction.
  Instance ``first_*``/``last_*`` describe the first/last collected steps,
  subject to ``kinematics_steps`` (``visible`` in the sample). Kinematic
  positions are mm and times ns. Showers have no last endpoint.

Aligned training products
~~~~~~~~~~~~~~~~~~~~~~~~~

All parsers share one event preparation. Coordinates are rasterized with the
file's ``voxel_size`` and an origin of zero, then shifted by an integer number
of cells to make event coordinates nonnegative. ``Meta`` records the inverse
transform in cm. Every spatial parser accepts ``voxel_size_mm`` (scalar or three-vector)
and ``origin_mm``; all spatial parsers in a dataset must use the same grid settings.
The identity-only ``event_info`` parser does not take grid arguments.
An explicit size is required when the file does not supply one. These settings
also permit rasterizing deposit-mode files. Upstream voxelization can use a
per-particle origin, so stored centers are rasterized rather than assumed to
already be integer coordinates on a common grid.

Overlapping contributions sum their energy. Label ownership follows upstream
priority (track, shower, delta, Michel, LE), then earliest deposit time, then
source row order. The same winning contribution selects semantic, fragment
and instance labels. LE deposits are retained. ``ClusterLabelData`` stores
fragment IDs and fragment-row associations, with instance IDs in ``group``.
Unprovided neutrino identity is left at ``-1``. No energy-to-charge conversion
is performed.

``coord_label`` is a separate, row-aligned ``particle_coords`` product for
GrapPA. Its ``TensorData`` schema has ``start`` and ``end`` coordinate groups,
plus ``time`` (ns) and ``shape`` features. Every stored particle gets a row,
including particles with no retained voxels. Instance rows use stored collected
step endpoints. Where these are missing, fragment rows use earliest/latest
collected deposit positions; instance rows use their non-LE deposit range,
or all deposits when that range is empty. These fallback positions are derived
from voxelized deposits, **not exact Geant4 step endpoints**. Non-track ends
repeat their starts. Rows without usable endpoints retain NaN coordinates and
times. Deposited rows have finite fallback coordinates; no creation/destruction
position is substituted. The same endpoint resolution populates parsed particle
``first_step``/``last_step`` fields.

PPN targets use finite stored instance starts and track ends only, exclude LE,
unknown classes and nuclear fragments, and omit points outside the event
image. They reference the complete event particle table but cannot replace
``coord_label`` for closest-primary selection or track orientation.

The particle parser fills PID, group/interaction-primary flags and ancestry
consistently with cluster supervision. PID and interaction-primary labels are
propagated from each fragment's host instance, preventing daughter-electron
votes from changing an instance's PID. A group primary is its representative;
an interaction primary is a host whose redirected stored parent is itself.
This is the file's stored-root convention, not an independently inferred
generator-primary definition. Host-instance interaction IDs label absorbed
fragments, whose own source interaction may differ. Stored interaction vertices
supply a common ``ancestor_position`` and cluster ``vertex`` for all members.
When unavailable, the stored ancestor's creation position is used; missing
ancestry remains NaN rather than indexing row ``-1``.

Individual ``pc_start:pc_end`` ranges do not cover all deposits in this file:
absorbed particles may have been omitted from the stored table. Fragment ranges
do partition the point cloud. SPINE assigns each fragment's summed dE once,
to its representative row, and zero to other rows. Truth particle construction
then sums these fragment contributions without double-counting. This energy
is independent of overlap-winner assignment; a truth object's voxel deposition
sum can differ from its attributed source-fragment energy.

Truth conversion
~~~~~~~~~~~~~~~~

Use the complete schema above with:

.. code-block:: yaml

   base:
     unwrap: true
   build:
     mode: truth
     units: cm
     fragments: true
     particles: true
     interactions: true

No model is required for truth-only construction. ``meta`` exposes the shared
``Meta`` at top level, so collation, unwrapping and ``BuildManager`` can convert
voxel coordinates to detector centimetres. Voxel centers use the half-cell
shift; particle endpoints and interaction vertices retain their continuous
physical positions. ``graph_label`` supplies directed stored-parent edges;
truth building contracts invisible ancestors and restores parent/child links
between retained particles. Without this product, those output genealogy links
are not restored. The standard SPINE HDF5 V2 writer/reader preserves these
products and associations; load truth objects through ``BuildManager`` to
reattach their long-form point arrays.

Unavailable neutrino identity, unknown PID and missing relationship IDs use
``-1``. Missing PDG codes default to SPINE's ``INVALID_PDG`` (0); the source
Geant4-parent PDG sentinel (-1) is retained when present. Absent creation
process names are empty strings. Missing positions, times and travel distance
are NaN. Neutrino interaction records, measured charge, generator-process
strings and exact endpoints of individual fragments cannot be recovered from
this file. The reader does not synthesize them. See the
:doc:`author-facing format report <supera_format_findings>` for measured format
limitations and proposed upstream changes.

.. toctree::
   :hidden:

   supera_format_findings

``groups`` and ``hit_labels/volume0/plane0`` (or another explicit plane path)
can be requested through ``SuperaReader(keys=...)``. They are never read by
the training parsers or by default. Even explicitly requested hit labels are
sliced by event offsets; no whole-plane label array is materialized.

Validation
~~~~~~~~~~

The synthetic tests cover sparse IDs, fragment/instance distinctions, LE,
semantic remapping, overlapping energy, empty events, multi-file selection,
absolute-range rebasing, pickle round trips and projected hit reads. Error-path
tests also cover invalid grids, corrupt relationships, nonfinite values,
missing compression support and endpoint filtering. The dedicated coverage
configuration requires 100% line and branch coverage of the Supera modules,
without excluding defensive or optional-dependency paths:

.. code-block:: bash

   PYTHONPATH=src pytest -q test/test_io/test_supera \
     --cov=src/spine --cov-config=test/test_io/test_supera/coverage.ini \
     --cov-report=term-missing

Run the real-file forward/backward test in the model runtime with:

.. code-block:: bash

   docker run --rm --platform linux/amd64 \
     -e PYTHONPATH=/workspace/src -e OMP_NUM_THREADS=1 \
     -e SPINE_SUPERA_SAMPLE=/workspace/supera_pixel_j000000-000004.h5 \
     -v "$PWD:/workspace" -w /workspace \
     ghcr.io/deeplearnphysics/spine:latest bash -lc \
     'pip install hdf5plugin && pytest -q -s test/test_io/test_supera'

The first sample event yields 25,537 unique voxels and 120 PPN targets.
The opt-in integration suite exercises these generic-chain components:

* UResNet segmentation and PPN forward/backward on the complete first event,
  with segmentation-stage cache publication and readback.
* Graph-SPICE fragmentation forward/backward on the same aligned voxel labels.
* Fragment-graph feature/target materialization and cache readback, followed by
  shower GrapPA primary/grouping and track GrapPA grouping forward/backward.
* Particle-graph feature/target materialization and cache readback, followed by
  interaction GrapPA PID, closest-primary, orientation and grouping
  forward/backward.
* Truth fragment, particle and interaction construction for events 0 and 1,
  followed by SPINE HDF5 V2 writing, reading and association rebuilding in cm.

GrapPA checks use two complete events with truth-defined input clusters and the
maintained generic configuration, reducing only hidden neural widths and
enabling closest-primary matching. They retain all configured objectives and
assert valid, nonzero supervised counts and gradients for each. These checks
validate each learned component and its cache contract; they do not measure
reconstruction quality or assert that an untrained end-to-end network produces
useful clusters. Synthetic tests additionally exercise sparse IDs larger than
the particle table, empty batch entries, missing endpoints, missing ancestors,
closest-primary matching and opposite track orientations across events.
The Supera modules retain 100% line and branch coverage. All exercised models
are available on ``main``; there is no dependency on UResNetMulti.

.. autosummary::
   :toctree: generated

   parse.supera.SuperaTensorParser
   parse.supera.SuperaSegmentationParser
   parse.supera.SuperaClusterLabelParser
   parse.supera.SuperaParticleParser
   parse.supera.SuperaParticlePointsParser
   parse.supera.SuperaParticleCoordsParser
   parse.supera.SuperaParticleGraphParser
   parse.supera.SuperaMetaParser
   parse.supera.SuperaSourceIDsParser
   parse.supera.SuperaEventInfoParser

Implementation modules
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   dataset.supera
   read.supera
   parse.supera.base
   parse.supera.cluster
   parse.supera.metadata
   parse.supera.particle
   parse.supera.tensor

Entry Filtering
---------------

File-aware filtering is performed before ordinary reader selection. The
generic scan and manifest APIs delegate physical inspection to a backend such
as :class:`filter.LArCVEntryInspector`,
:class:`filter.HDF5EntryInspector` or
:class:`filter.CacheEntryInspector`.

.. autosummary::
   :toctree: generated

   filter.EntryInspector
   filter.hdf5.HDF5EntryInspector
   filter.hdf5.CacheEntryInspector
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
     provider: hdf5
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
     provider: hdf5
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
       provider: hdf5
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

Legacy files may omit a field whose interpretation is known from provenance.
Use ``object_defaults`` to supply that field without overwriting values in
newer files which store it explicitly. For example, force missing neutrino
interaction schemes to LArSoft as follows:

.. code-block:: yaml

   io:
     reader:
       provider: hdf5
       file_keys: legacy_output.h5
       object_defaults:
         Neutrino:
           interaction_scheme: larsoft

The same option belongs under ``io.loader.dataset`` when an HDF5 dataset is
used through a data loader. Defaults are restricted to fixed-width scalar data
class fields and apply only when the field is absent from the stored schema;
they do not reinterpret an explicitly stored sentinel value.

The LArCV neutrino parser accepts ``larsoft``, ``genie``, ``gibuu``, ``nuwro``,
and ``neut`` for ``interaction_scheme``. The selected scheme controls how
``interaction_mode`` and ``interaction_type`` are resolved to named enum
members; it does not translate codes between generators. For the alternative
generators, modes use normalized GENIE scattering categories for convenient
cross-generator comparisons. GiBUU and NEUT interaction types use the process
IDs published in their NuHepMC metadata, while NuWro types use the simplified
NEUT-style channel ID written by ``nuwro2rootracker``.

Datasets
--------

The dataset layer bridges low-level readers and parser logic into map-style
event sources. A dataset can be nested below ``io.loader`` when a model needs
batched and collated products, or selected directly as ``io.dataset`` for
event-level construction, post-processing, analysis, and writing.

Direct dataset traversal does not require PyTorch and does not collate or
unwrap its products. For example, a LArCV-to-HDF5 conversion can reuse the
same parser schema as a loader-backed configuration:

.. code-block:: yaml

   base:
     iterations: -1

   io:
     dataset:
       provider: larcv
       file_keys: /path/to/input.root
       schema:
         particles:
           provider: particle
           particle_event: particle_pcluster
           cluster_event: cluster3d_pcluster
         meta:
           provider: meta
           sparse_event: sparse3d_pcluster
     writer:
       provider: hdf5
       file_name: /path/to/output.h5

   build:
     mode: truth
     units: cm
     fragments: true
     particles: true
     interactions: true

Exactly one of ``io.reader``, ``io.dataset``, or ``io.loader`` must be
configured. A direct ``JointDataset`` uses deterministic sequential pairing:
the primary source defines the traversal length and primary entry ``i`` is
overlaid with secondary entry ``i % len(secondary)``. Loader-backed joint
datasets retain their configured random, probabilistic, or sequential sampler
behavior.

SPINE cache repositories have their own dataset type and can also be paired
with raw input through the mixed dataset.

.. autosummary::
   :toctree: generated

   dataset.LArCVDataset
   dataset.CacheDataset
   dataset.HDF5Dataset
   dataset.SuperaDataset
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
       provider: cache
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
         provider: cache
         path: /path/to/train.spine-cache
         stage: deghosting
     writer:
       provider: cache
       stage: fragmentation

Mixed raw/cache training remains a normal mixed dataset. The child roles are
explicit: ``primary`` selects the authoritative dataset implementation, while
``cache`` is a manifest-backed SPINE cache. Storage-format aliases such as
``larcv`` and ``hdf5`` are intentionally not part of the mixed-dataset
contract:

.. code-block:: yaml

   dataset:
     provider: mixed
     primary:
       provider: larcv
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
       provider: cache
       path: /path/to/train.spine-cache
       stage: deghosting
       parallel: true
       expected_sources: 400

Every task in one submission attempt must also receive the same
``SPINE_CACHE_PUBLICATION_ID`` environment variable. The orchestration layer
should generate a fresh UUID for each cache-stage submission and propagate it
to every array task and scheduler chunk. This attempt identity is intentionally
separate from experiment YAML and prevents shards from different retries or
configurations from being combined.

Before submitting those workers, register the attempt atomically::

   spine-cache begin /path/to/train.spine-cache deghosting \
     --publication-id 4c277...

Omitting ``--publication-id`` generates one and prints it for the launcher to
capture. Registration is the publication fence: workers may contribute only
under the currently registered ID. Beginning a newer attempt retires an older
incomplete candidate before any new worker starts, so delayed tasks from the
old attempt are rejected rather than allowed to reclaim the stage.

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
safe to retry within one publication ID. The first contribution carrying a new
publication ID discards an incomplete candidate left by an older attempt. When
the final expected source arrives, the manifest atomically activates the
assembled replacement and invalidates the old stage plus all of its
descendants. No merge or finalization job is required.

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
