Configuration Reference
=======================

SPINE configuration follows the pipeline from data input to analysis. This
page maps each YAML block to the API object that validates and consumes it.
Use :doc:`config_loader` for includes and overrides; use the links below for
accepted parameters, defaults, and output behavior.

Configuration Lifecycle
-----------------------

The configuration passed to the driver is the result of several ordered
steps:

1. :func:`spine.config.load_config_file` resolves includes, removals,
   overrides, environment variables, and metadata constraints.
2. ``spine --inference`` optionally applies
   :func:`spine.config.to_inference_config`.
3. Explicit CLI source, output, resource, checkpoint, and ``--set`` overrides
   are applied.
4. :func:`spine.config.normalize_config` migrates supported legacy forms.
5. :class:`spine.driver.Driver` and each stage manager validate and construct
   their owned blocks.

Use ``spine-config dump CONFIG --output resolved.yaml`` to inspect steps one
and four without downloading ``!download`` resources. The startup log is the
authoritative record after all CLI overrides have been applied.

Top-level Contract
------------------

.. list-table:: Top-level configuration contract
   :header-rows: 1
   :widths: 17 21 62

   * - Block
     - Requirement
     - Owner and constraints
   * - ``base``
     - Required by runtime
     - Driver and launcher controls. Unknown keys are rejected. The CLI creates
       an empty block if omitted, but direct :func:`spine.main.run` requires it.
   * - ``io``
     - Required
     - :class:`spine.io.manager.IOManager`. A run needs a reader and/or loader;
       models require a loader.
   * - ``geo``
     - Optional
     - :class:`spine.geo.manager.GeoManager`; required by geometry-dependent
       calibration, construction, or post-processing.
   * - ``model``
     - Optional
     - :class:`spine.model.manager.ModelManager`; requires ``io.loader`` and the
       full model runtime.
   * - ``train``
     - Optional
     - Selects training mode and requires ``model``. The legacy ``base.train``
       form is normalized with a migration warning.
   * - ``validation``
     - Optional, training only
     - Requires ``train``, ``io.loader``, and a checkpoint cadence.
   * - ``build``
     - Optional
     - :class:`spine.construct.manager.BuildManager`; reconnects or creates
       fragment, particle, and interaction objects.
   * - ``post``
     - Optional
     - Ordered registry-backed processors. Required products must exist before
       each processor runs.
   * - ``ana``
     - Optional
     - Ordered registry-backed analyses executed after post-processing.

Top-level blocks
----------------

``base``
   Driver execution controls such as entry ranges, seeding, logging,
   distributed execution, and device selection. See
   :meth:`spine.driver.Driver.initialize_base`.

``geo``
   Detector geometry selected by detector name/tag or supplied component
   configuration. See :class:`spine.geo.manager.GeoManager` and the
   structured :class:`spine.geo.base.Geometry` reference. Nested ``tpc``,
   ``optical``, and ``crt`` blocks are documented under :doc:`api/geo`.

``io``
   Input, output, datasets, parsers, batching, sampling, and augmentation.
   Start with :class:`spine.io.manager.IOManager`, then follow the selected
   reader, writer, dataset, parser, or augmenter on :doc:`api/io`.

``model``
   A registered network/loss pair. ``model.name`` selects the top-level
   model and ``model.modules`` contains its nested constructor mappings.
   See :class:`spine.model.manager.ModelManager` and :doc:`api/model`.

``train``
   Optimizer, learning-rate scheduler, checkpoint, and iteration controls
   consumed by :class:`spine.model.manager.ModelManager`.

``validation``
   Checkpoint-bound validation, monitored metrics, early stopping, and
   best-checkpoint selection. See
   :class:`spine.model.validation.ValidationManager`,
   :class:`spine.model.validation.EarlyStopping`, and
   :class:`spine.model.validation.BestCheckpoint`.

``build``
   Construction of fragment, particle, and interaction objects. See
   :class:`spine.construct.manager.BuildManager` and :doc:`api/build`.

``post``
   Ordered post-processing entries. Each key selects a registered processor;
   the nested mapping is passed to its constructor. See the complete
   processor list in :doc:`api/post`.

``ana``
   Analysis scripts run after post-processing. Each key selects a registered
   analyzer and its nested mapping supplies constructor parameters. See
   :doc:`api/ana`.

How a named entry maps to Python
--------------------------------

For registry-backed sections, the YAML key or ``name`` value maps to a
class's ``name`` attribute. Constructor parameters become keys in that
entry. For example:

.. code-block:: yaml

   post:
     vertex:
       run_mode: both
       use_primaries: true
       touching_threshold: 2.0

The ``vertex`` key selects
:class:`spine.post.reco.VertexProcessor`. Its constructor reference is the
authoritative list of accepted settings, including inherited post-processing
controls such as ``run_mode`` and truth-coordinate modes.

Nested registries follow the same rule. A parser under
``io.loader.dataset.schema``, an augmenter under
``io.loader.dataset.augment``, and a calibrator nested in a calibration
processor each resolve to the class listed in their package API page.

Finding a setting
-----------------

1. Identify the top-level block and open its API page above.
2. Follow the class matching the configured key or ``name``.
3. Read its constructor parameters for accepted keys, defaults, and units.
4. Follow linked data classes for fields produced or modified by the component.

The :doc:`api/modules` index is the exhaustive implementation reference when
an object is not configuration-selected. It complements the focused pages; it
is not the recommended starting point for configuring a pipeline.

Validation Scope
----------------

SPINE validates settings at the layer that owns them. Loading YAML proves that
composition syntax and metadata are valid; manager construction validates
component names and constructor settings; execution validates event-dependent
product contracts. A production preflight therefore includes all three:

.. code-block:: bash

   spine-config dump candidate.yaml --output resolved.yaml
   spine -c candidate.yaml --source representative.root --iterations 1

Use :doc:`workflows` for complete commands and :doc:`operations` for the
production acceptance checklist.
