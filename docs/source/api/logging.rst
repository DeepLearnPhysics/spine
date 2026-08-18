Logging Module
==============

The :mod:`spine.logging` package owns human-readable logging, structured CSV
metric logging, and run-level log coordination. :class:`CSVLogger` writes
analysis and training metrics; it is not a generic SPINE event-output writer.

.. currentmodule:: spine.logging

.. autosummary::
   :toctree: generated

   CSVLogger
   LogManager
   MainProcessFilter
   configure_rank_logging
   logger
