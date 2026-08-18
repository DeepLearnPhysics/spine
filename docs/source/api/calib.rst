Calibration Module
==================

The :mod:`spine.calib` package contains detector-response corrections and the
manager that applies them. Calibration implementations are detector-facing
operations; reusable particle-physics calculations live in
:mod:`spine.physics`.

.. currentmodule:: spine.calib

Core interface
--------------

.. autosummary::
   :toctree: generated

   CalibrationManager

Calibration implementations
---------------------------

.. autosummary::
   :toctree: generated

   constant
   database
   field
   function
   gain
   lifetime
   recombination
   response
   smearing
   transparency
