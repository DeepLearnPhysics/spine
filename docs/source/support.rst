Supported Interfaces And Compatibility
======================================

SPINE is a production tool, but not every importable implementation helper is
a compatibility promise. This page identifies the interfaces intended for
users and the boundaries that must be pinned for reproducible operation.

Supported User-Facing Surface
-----------------------------

The following are the primary supported entry points for a released version:

- the ``spine`` and ``spine-config`` command-line interfaces;
- :class:`spine.driver.Driver` and :func:`spine.main.run`;
- :mod:`spine.config` loading, composition, and inference transforms;
- top-level configuration blocks and registry-backed names documented in
  :doc:`configuration`;
- public data products documented in :doc:`data_model` and :doc:`api/data`;
- focused package references linked directly from :doc:`api/index`.

The :doc:`api/modules` page deliberately exposes lower-level implementation
modules for developers and advanced debugging. Presence in that exhaustive
index alone does **not** make a symbol a stable application interface. Prefer a
focused package page or top-level import when one exists.

Version And Reproducibility Boundary
------------------------------------

A SPINE release consists of source/package version, maintained
configurations, and its matching runtime container. Pin all three together for
production. ``latest`` is convenient for evaluation but is not a reproducible
runtime identifier; record the release tag and image digest.

The project preserves compatibility where practical and emits migration
warnings for selected legacy forms, such as ``base.train``. It does not promise
that every lower-level module path or undocumented configuration combination
will remain unchanged across releases. Qualify a new release with the process
in :doc:`operations` and review GitHub release notes before adoption.

Configurations
--------------

Maintained files under ``config/`` are executable contracts and are loaded by
CI in the complete runtime. User configurations should start from those files,
use the documented include/override language, and be archived in resolved form.

Registry names and constructor parameters on focused API pages define accepted
component settings for that release. Unknown top-level ``base`` keys are
rejected; deeper validation occurs when the owning manager or component is
constructed. Therefore, successful YAML parsing is not equivalent to a valid
runtime configuration.

Checkpoints And Outputs
-----------------------

New checkpoints contain a format version, runtime manifest, normalized
configuration, dataset provenance, and checksum. Legacy checkpoints may load
with reduced resume guarantees. Always inspect warnings and use strict
``--resume`` when continuation rather than weights-only initialization is
required.

HDF5 files store a SPINE-producing version separately from their physical
schema. Readers provide compatibility handling for supported layouts, but
production consumers should retain the producing release and test representative
files before upgrading.

Runtime Dependency Levels
-------------------------

.. list-table:: Runtime support levels
   :header-rows: 1
   :widths: 25 34 41

   * - Environment
     - Intended use
     - Support expectation
   * - Release-tagged container
     - Training, inference, full reconstruction
     - Primary production environment with the compatible compiled stack
   * - Core ``pip install spine``
     - Configuration, HDF5 inspection, CPU analysis and post-processing
     - Tested without optional ML dependencies
   * - ``spine[viz]``
     - Plotting and event display
     - Core environment plus supported visualization libraries
   * - Manually assembled ML stack
     - Specialized development
     - Advanced path; the operator owns CUDA/PyTorch/PyG/Minkowski/LArCV
       compatibility

Reporting Compatibility Problems
--------------------------------

Report the exact SPINE release or commit, container digest, Python/platform,
dependency information from ``spine --info``, resolved configuration, input
format, and complete traceback. For checkpoint or HDF5 issues, include the
producing SPINE version and manifest without sharing private data.
