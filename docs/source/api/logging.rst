Logging Module
==============

The :mod:`spine.logging` package owns human-readable logging, structured CSV
metric logging, and run-level log coordination. :class:`CSVLogger` writes
analysis and training metrics; it is not a generic SPINE event-output writer.

Distributed progress output
---------------------------

Stdout still reports every iteration by default (base.log_step: 1).
Distributed runs print one compact line on global rank zero, with the
minimum/maximum iteration time and peak allocated GPU memory across ranks.
Loss and accuracy are arithmetic means of the per-rank scalar values. Training
metrics have already been reduced globally; validation progress averages the
current batch's rank metrics, while the final validation summary remains the
authoritative result for the complete pass.

For diagnostic per-rank tables, set base.log_ranks: true (or pass
--set base.log_ranks=true). This also applies to on-the-fly validation.
Neither option changes the frequency or contents of per-rank CSV records.

Compact summaries gather a fixed-size numeric tensor rather than serialized
Python strings. Neither compact nor detailed logging inserts a separate barrier
before its collective. This reduces presentation work but does not eliminate
logging communication or guarantee a throughput improvement. Benchmark total
elapsed time: iteration timers exclude the subsequent logging work.

Under an external launcher such as torchrun, the CLI banner and startup
header are printed only by global rank zero. Warnings and errors remain visible.

.. currentmodule:: spine.logging

.. autosummary::
   :toctree: generated

   CSVLogger
   LogManager
   MainProcessFilter
   configure_rank_logging
   logger
