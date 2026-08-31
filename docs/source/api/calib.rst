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

These are the classes accepted in a calibration chain.  Follow a class link
to review its configurable constructor parameters and defaults; the shorter
module links below retain the implementation-level API.

.. autosummary::
   :toctree: generated

   field.FieldCalibrator
   gain.GainCalibrator
   lifetime.LifetimeCalibrator
   recombination.RecombinationCalibrator
   response.ResponseCalibrator
   smearing.SmearingCalibrator
   transparency.TransparencyCalibrator

Supporting configuration objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   field.FieldMap

Implementation modules
~~~~~~~~~~~~~~~~~~~~~~

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
