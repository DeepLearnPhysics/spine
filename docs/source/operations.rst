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
