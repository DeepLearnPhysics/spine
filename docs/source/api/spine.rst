SPINE Core
==========

The top-level :mod:`spine` package exposes the main entry points for running a SPINE job. In practice, most users interact with :class:`spine.driver.Driver`, which coordinates the full reconstruction chain from configuration loading through output writing.

.. currentmodule:: spine

.. automodule:: spine
   :no-members:

Core Entry Points
-----------------

The core interfaces are intentionally small:

- ``driver`` defines the high-level execution pipeline used by training, inference, and analysis jobs
- ``main`` provides the command-line entry point used by the ``spine`` executable

.. autosummary::
   :toctree: generated

   driver
   main
