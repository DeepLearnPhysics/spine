Math Module
===========

The ``spine.math`` module provides the Numba-accelerated numerical kernels and geometry helpers used internally across SPINE. These routines support clustering, neighborhood queries, distance calculations, decomposition, and graph-style operations that appear throughout reconstruction and analysis.

.. currentmodule:: spine.math

.. automodule:: spine.math
   :no-members:

Core Modules
------------

This package is an implementation layer for the reconstruction stack, not the main user-facing entry point of SPINE. It is documented here because many higher-level modules depend on it.

.. autosummary::
   :toctree: generated

   base
   cluster
   decomposition
   distance
   graph
   linalg
   metrics
   neighbors
