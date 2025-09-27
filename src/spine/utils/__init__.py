"""Utility functions and tools used across the SPINE package.

This module contains reusable utility functions organized into several categories:

**Core Utilities:**
- `conditional`: Conditional imports for optional dependencies (ROOT, LArCV, MinkowskiEngine, PyTorch)
- `config`: Configuration file parsing and validation
- `factory`: Generic factory pattern implementations
- `logger`: Logging utilities and configuration
- `globals`: Global constants and enumerations

**Decorators:**
- `docstring`: Docstring inheritance utilities for class hierarchies
- `jit`: JIT compilation decorators using Numba for performance optimization

**Mathematical & Scientific:**
- `cluster`: CNN clustering algorithms and utilities
- `gnn`: Graph Neural Network utilities (network operations, clustering, voxel processing)
- `geo`: Geometry calculations and transformations
- `particles`: Particle physics calculations and utilities
- `pid`: Particle identification algorithms
- `tracking`: Track reconstruction algorithms
- `vertex`: Vertex reconstruction utilities

**PyTorch Utilities:**
- `torch`: PyTorch utilities organized into submodules:
  - `torch.runtime`: Tensor operations, CUDA memory management, distributed operations
  - `torch.devices`: CUDA device configuration and distributed training setup
  - `torch.training`: Optimizer and learning rate scheduler factories
  - `torch.scripts`: Utility scripts and PyTorch function extensions (e.g., cdist_fast)
  - `torch.adabound`: Custom AdaBound and AdaBoundW optimizer implementations

**Hardware & Performance:**
- `energy_loss`: Energy loss calculations and CSDA tables
- `optical`: Optical detector simulation and analysis
- `calib`: Calibration utilities and corrections

**Analysis Tools:**
- `ppn`: Point Proposal Network utilities
- `match`: Truth matching algorithms
- `metrics`: Evaluation metrics and statistics
- `inference`: Model inference utilities
- `stopwatch`: Performance timing utilities

**Data Processing:**
- `unwrap`: Data unwrapping and processing utilities
- `weighting`: Sample weighting and balancing

Most utilities are designed to be framework-agnostic where possible, with PyTorch-specific
functionality clearly separated and conditionally imported.
"""
