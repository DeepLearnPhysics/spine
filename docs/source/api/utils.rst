Utilities
=========

The ``spine.utils`` package collects the cross-cutting helpers that support the driver, model, reconstruction, and analysis layers. It includes optional-dependency handling, logging, factories, reconstruction-specific utilities, and numerical helpers that do not belong to a single pipeline stage.

.. currentmodule:: spine.utils

.. automodule:: spine.utils
   :no-members:

Core Utilities
--------------

.. autosummary::
   :toctree: generated

   conditional
   docstring
   factory
   globals
   logger
   stopwatch

Physics And Reconstruction Utilities
------------------------------------

.. autosummary::
   :toctree: generated

   energy_loss
   match
   metrics
   optical
   particles
   pid
   ppn
   tracking
   vertex
   weighting

Numerical And Graph Utilities
-----------------------------

.. autosummary::
   :toctree: generated

   cluster
   gnn
   inference
   jit
   unwrap
   torch
