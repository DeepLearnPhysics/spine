Pipeline Architecture
=====================

SPINE is organized as a sequence of explicit processing stages coordinated by
:class:`spine.driver.Driver`. A configuration may omit stages it does not
need, but the ownership boundaries remain the same.

Execution Flow
--------------

``spine.io``
   Reads detector files, parses records into :mod:`spine.data` products,
   batches model inputs, and writes event outputs.

``spine.model``
   Owns trainable reconstruction code, checkpoint handling, loss evaluation,
   and model-specific transformations. The released runtime container is the
   supported environment for this stage.

``spine.construct``
   Converts model products into the fragment, particle, and interaction object
   hierarchy.

``spine.post``
   Applies reconstruction and truth post-processors, including detector
   matching and derived physical quantities.

``spine.ana``
   Runs analysis scripts over completed event products and records tabular
   results through :mod:`spine.logging`.

Across those stages, :mod:`spine.geo`, :mod:`spine.calib`,
:mod:`spine.cluster`, :mod:`spine.math`, and :mod:`spine.physics` provide
shared domain services. :mod:`spine.utils` is intentionally limited to
cross-layer helpers that do not belong to one of those domains.

Driver Lifecycle
----------------

The driver validates and normalizes the supplied configuration, initializes
the configured managers, and processes entries in the requested range. For an
interactive workflow, :meth:`spine.driver.Driver.process` returns one event's
data products. For a complete configured job, :meth:`spine.driver.Driver.run`
owns the iteration loop, logging, and configured output handling.

The command-line executable uses the same path:

.. code-block:: bash

   spine --config config/full_chain/full_chain_test.yaml --source data.root

Command-line input, output, checkpoint, and ``--set`` options override the
corresponding configuration values before the driver is constructed. See
:doc:`config_loader` for the configuration language and :doc:`quickstart` for
container commands.

Choosing An API Level
---------------------

- Use the ``spine`` command for reproducible training, inference, conversion,
  and analysis jobs.
- Use :class:`spine.driver.Driver` when a Python application must process or
  inspect events directly.
- Use individual package APIs when developing a pipeline stage or consuming
  already-produced SPINE data.

The API reference mirrors these ownership boundaries so that a moved symbol
has one canonical documented location.
