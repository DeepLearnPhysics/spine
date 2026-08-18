"""Small tools shared across otherwise independent SPINE packages.

Only cross-layer infrastructure belongs here: optional-dependency adapters,
JIT/docstring helpers, timing and manager infrastructure, and PyTorch runtime
adapters. ``ghost``, ``optical`` and the minimal PPN prediction interface are
explicit domain exceptions because each bridges multiple top-level packages.

Numerical algorithms live in :mod:`spine.math`, reconstruction calculations in
:mod:`spine.physics`, cluster operations in :mod:`spine.cluster`, configuration
tools in :mod:`spine.config`, and implementation-specific helpers beside their
owning IO, model, post-processing, or analysis code.
"""
