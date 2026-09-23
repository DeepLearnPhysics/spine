Production Operations
=====================

SPINE is a production reconstruction tool, so a successful process exit is
only one part of a valid run. This page defines the operational evidence to
retain and the checks to perform before scaling out.

Preflight Checklist
-------------------

1. Select a release tag and use the matching
   ``ghcr.io/deeplearnphysics/spine:<release>`` image. Do not use ``latest``
   for a result that must be reproduced later.
2. Run ``spine --info`` inside the execution environment and retain its output.
3. Resolve the configuration with ``spine-config dump`` and review the result,
   especially inputs, checkpoint paths, writer keys, detector geometry,
   calibration sources, post-processors, and analysis scripts.
4. Record immutable input identifiers or file manifests. Shell globs and
   mutable directory listings are convenient launch inputs but insufficient
   provenance.
5. Verify checkpoint checksums before launch. Retain the checkpoint manifest
   and source revision reported by :func:`spine.model.inspect_checkpoint`.
6. Run a small representative subset with ``--iterations`` and inspect its
   logs and output schema before scaling to the complete dataset.
7. Confirm output and log locations have sufficient capacity and the intended
   overwrite policy.

Artifacts To Retain
-------------------

Retain these items together for each production campaign:

- SPINE release tag and container image digest;
- output of ``spine --info``;
- fully resolved YAML configuration;
- exact CLI invocation or batch submission script;
- input manifest and detector/calibration database versions;
- model checkpoint plus ``.sha256`` and manifest;
- stdout/stderr, CSV logs, validation logs, and TensorBoard events;
- output files and their checksums;
- scheduler job identifier, host/GPU allocation, and distributed world size.

File-aware Entry Filtering
--------------------------

Pathological raw events should be rejected before a driver or parser loads
their products. ``spine-filter`` first records requested per-entry product
sizes in reusable, source-fingerprinted counter files, then builds one compact
manifest whose rejected indexes are local to each source file::

   spine-filter scan \
     --config filter.yaml \
     --source-list raw-files.txt \
     --cache-dir filter-cache \
     --workers 4

   spine-filter build \
     --config filter.yaml \
     --source-list raw-files.txt \
     --cache-dir filter-cache \
     --output accepted.yaml \
     --output-source-list accepted-files.txt

The initial LArCV configuration language deliberately supports only product
sizes and exclusive upper bounds. This example accepts an event exactly when
``sparse3d_reco_count < 500000`` while retaining two diagnostic counts::

   input:
     name: larcv

   measurements:
     sparse3d_reco: {kind: product_size}
     sparse3d_pcluster: {kind: product_size}
     particle_pcluster: {kind: product_size}

   filters:
     sparse3d_reco: {max_count: 500000}

Valid scan records are reused; changed source size, modification time, backend,
inspector version, or measurement specification causes a rescan. ``--force``
requests an unconditional rescan. Both counter records and final artifacts are
published atomically.

Apply the resulting manifest to the raw LArCV reader::

   spine -c train.yaml \
     --source-list raw-files.txt \
     --entry-filter accepted.yaml \
     --val-entry-filter accepted-test.yaml

The manifest may describe the complete dataset while a scheduler task reads
only one listed source. Missing files, changed fingerprints, and entry-count
mismatches are hard errors. The filter establishes eligibility before normal
selection, so a configured ``entry_fraction_range`` splits the surviving
sequence into adjacent, non-overlapping partitions. Numeric limits and skips
also operate on survivors; explicit entry and run/event lists retain their
source-domain meaning and are intersected with eligibility.

An unfiltered flat or staged HDF5 cache can apply the same LArCV manifest
without rescanning its cached products::

   spine -c train-from-cache.yaml \
     --source-list cache-files.txt \
     --entry-filter accepted.yaml

SPINE maps each physical cache entry back to the manifest using its persisted
``source_file_name``, ``source_file_size``, ``source_file_mtime_ns`` and
``source_file_entry_index`` provenance. Missing or ambiguous provenance is a
hard error; SPINE never assumes that physical HDF5 order equals raw LArCV
order.

Cached products can also be inspected directly. This is useful when a product
created during cache construction, rather than a raw LArCV product, determines
whether an entry is safe to train on. For a logical cache repository, configure
the ``cache`` backend and name the cached product explicitly:

.. code-block:: yaml

   input:
     name: cache

   measurements:
     shower_edges:
       kind: product_size
       product: fragment_graph_shower_edge_index

   filters:
     shower_edges:
       max_count: 1600001

The existing ``max_count`` predicate is an exclusive upper bound. The value
above therefore accepts edge counts through 1,600,000 and rejects larger
graphs. If the same product name is published by more than one stage, add the
owning stage to the measurement:

.. code-block:: yaml

   measurements:
     shower_edges:
       kind: product_size
       product: fragment_graph_shower_edge_index
       stage: fragment_graph

Scan and build against the logical repository, not its private shard paths:

.. code-block:: bash

   spine-filter scan \
     --config shower-edge-filter.yaml \
     --source /path/to/train.spine-cache \
     --cache-dir shower-edge-scan

   spine-filter build \
     --config shower-edge-filter.yaml \
     --source /path/to/train.spine-cache \
     --cache-dir shower-edge-scan \
     --output accepted-cache.yaml \
     --output-source-list accepted-cache-sources.txt

Then configure the generated manifest on the cache dataset:

.. code-block:: yaml

   io:
     loader:
       dataset:
         name: cache
         path: /path/to/train.spine-cache
         entry_filter: accepted-cache.yaml

The inspector resolves the product's published stage and reads only its V2
``event_offsets`` metadata. It does not deserialize the edge index or edge
features. The cache reader computes one eligibility sequence before ordinary
entry selection and applies it identically to every requested stage, including
stages which do not own the measured product. This keeps graph inputs and
cached targets aligned before the sampler is constructed.

For an ordinary flat SPINE HDF5 cache, use ``input.name: hdf5`` and pass its
files to the same scan and build commands. Version 2 files are measured from
``event_offsets``; legacy version 1 files are measured from region-reference
selection extents without reading their payloads. A native HDF5 manifest is
tied to the exact files scanned and cannot be applied to a different physical
cache through source provenance.

A compact downstream cache which already contains only accepted entries should
not reapply the raw-data manifest.

For a mixed LArCV/HDF5 dataset, ``cache_entry_domain`` controls how the
filtered raw selection maps onto the cache. ``filtered`` denotes a compact
cache containing only eligible entries, while ``source`` denotes a cache that
retains the original unfiltered entry domain. The default ``auto`` mode infers
these layouts from their cardinalities and fails if neither is consistent.

The normalized configuration embedded in checkpoints is valuable evidence but
does not replace the launch command: CLI overrides are applied before the
driver starts and are reflected in startup logs.

Output Acceptance
-----------------

For a new configuration or release, acceptance should include more than file
existence:

- expected entry count and run/subrun/event identity;
- expected HDF5 keys and object families;
- finite coordinates, energies, scores, and derived quantities where required;
- consistency between object indices and their point/deposition products;
- stable aggregate physics or reconstruction metrics against an approved
  reference sample;
- explicit review of any warnings, skipped entries, or checksum failures.

Use the repository's output comparison utilities or experiment-specific
regression checks for numerical acceptance. Tolerances must be chosen for the
observable and hardware environment; byte identity is not a generally valid
GPU acceptance criterion.

Distributed Runs
----------------

``base.world_size`` is the total process count. For a single-node run, the CLI
can spawn one process per device. For multi-node execution, the external
launcher supplies ``RANK`` and ``WORLD_SIZE``; an explicit ``--world-size``
must agree with that environment. Distributed training requires DDP.

Rank zero owns primary logging, TensorBoard output, and shared checkpoint
publication. Input sharding, output naming, and shared filesystem visibility
must be tested on the target scheduler before a large launch. Resume with the
same world size when exact rank-local loader and RNG restoration matters.

Failure And Restart Policy
--------------------------

Treat checksum failures, missing required products, schema mismatches, and
configuration errors as hard failures. Do not work around them by disabling
validation without understanding the incompatibility.

For training, resume from the newest checkpoint whose checksum verifies and
whose manifest matches the intended run. ``--resume`` requests strict state
restoration; ``--no-resume`` deliberately begins a new optimization history
from loaded weights and should be recorded as a new run.

For inference, decide whether output files are atomic at the campaign level.
SPINE can split output by input file, but the orchestration system remains
responsible for detecting completed shards, quarantining partial files, and
preventing two jobs from writing the same destination.

Release Qualification
---------------------

Before adopting a new SPINE release for production:

1. Build the warning-strict documentation and review the API/configuration
   changes.
2. Run the maintained configuration-contract tests in the released runtime.
3. Run a representative end-to-end regression with the production detector,
   databases, checkpoint, and input format.
4. Compare accepted physics/reconstruction metrics with the currently
   qualified release.
5. Archive the qualification configuration, manifests, outputs, comparison
   report, and approval decision.

See :doc:`support` for the supported interface boundary and compatibility
expectations.
