"""Utility functions and tools used across the SPINE package.

This module contains reusable utility functions organized into several categories:

Core utilities:

- ``conditional`` for optional imports such as ROOT, LArCV, MinkowskiEngine, and PyTorch.
- ``factory`` and ``logger`` for application infrastructure.
- ``globals`` for deprecated shared-constant compatibility.

Scientific utilities:

- ``cluster`` and ``gnn`` for clustering and graph operations.
- ``particles``, ``pid``, ``tracking``, and ``vertex`` for reconstruction helpers.
- ``energy_loss`` and ``optical`` for detector-physics calculations.

Developer utilities:

- ``docstring`` for docstring inheritance.
- ``jit`` for Numba-backed acceleration.
- ``stopwatch`` for timing and profiling.
- ``unwrap`` and ``weighting`` for data processing.

Most utilities are designed to be framework-agnostic where possible, with PyTorch-specific
functionality clearly separated and conditionally imported.
"""
