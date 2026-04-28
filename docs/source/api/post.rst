Post-processing Module
======================

The ``spine.post`` module refines the objects and predictions coming out of model execution and construction. It collects configurable processors for physics cleanup, detector matching, calibration-aware corrections, and truth-aware bookkeeping.

.. currentmodule:: spine.post

.. automodule:: spine.post
   :no-members:

Module Index
------------

This package sits after construction in the standard pipeline and before analysis or writing, making it the main place for converting raw reconstruction output into analysis-ready quantities.

.. autosummary::
   :toctree: generated

   base
   manager
   factories
   template
   crt
   optical
   reco
   trigger
   truth
