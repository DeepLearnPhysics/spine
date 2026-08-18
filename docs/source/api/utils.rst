Utilities
=========

The :mod:`spine.utils` package is deliberately limited to small tools shared
across otherwise independent SPINE packages. Numerical, clustering, physics,
configuration, logging, and package-owned helpers have dedicated homes.

.. currentmodule:: spine.utils

.. automodule:: spine.utils
   :no-members:

Core Utilities
--------------

.. autosummary::
   :toctree: generated

   conditional
   docstring
   jit
   manager
   stopwatch

Cross-layer domain adapters
---------------------------

These are explicit exceptions to the generic-infrastructure rule because each
bridges more than one top-level package.

.. autosummary::
   :toctree: generated

   ghost
   optical
   ppn

Optional runtime adapters
-------------------------

.. autosummary::
   :toctree: generated

   torch
