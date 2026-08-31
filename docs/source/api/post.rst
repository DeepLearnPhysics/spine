Post-processing Module
======================

The ``spine.post`` module refines the objects and predictions coming out of model execution and construction. It collects configurable processors for physics cleanup, detector matching, calibration-aware corrections, and truth-aware bookkeeping.

.. currentmodule:: spine.post

.. automodule:: spine.post
   :no-members:

Configuration
-------------

This package sits after construction in the standard pipeline and before analysis or writing, making it the main place for converting raw reconstruction output into analysis-ready quantities.

Each key below is the Python class behind a ``post:`` configuration entry.
The class page shows the accepted constructor parameters, their defaults, and
the processor's methods.  In YAML, select a processor with its ``name`` class
attribute (for example, ``vertex`` for :class:`reco.VertexProcessor`):

.. code-block:: yaml

   post:
     vertex:
       run_mode: both
       use_primaries: true
       touching_threshold: 2.0

Reconstruction processors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   reco.CalorimetricEnergyProcessor
   reco.CalibrationProcessor
   reco.CalorimetricDirectionProcessor
   reco.CathodeCrosserProcessor
   reco.TrackClusterer
   reco.DirectionProcessor
   reco.ContainmentProcessor
   reco.FiducialProcessor
   reco.ParticleShapeLogicProcessor
   reco.ParticleThresholdProcessor
   reco.ParticleNeutrinoLogicProcessor
   reco.InteractionTopologyProcessor
   reco.MCSEnergyProcessor
   reco.PIDTemplateProcessor
   reco.TrackExtremaProcessor
   reco.PPNProcessor
   reco.ShowerParametricEnergyProcessor
   reco.ShowerConversionDistanceProcessor
   reco.ShowerStartMergeProcessor
   reco.ShowerStartCorrectionProcessor
   reco.SourceAssigner
   reco.ParticleDEDXProcessor
   reco.ParticleStartStraightnessProcessor
   reco.ParticleSpreadProcessor
   reco.CSDAEnergyProcessor
   reco.VertexProcessor

Detector and truth processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   crt.CRTMatchProcessor
   optical.FlashMatchProcessor
   trigger.TriggerProcessor
   truth.ChildrenProcessor
   truth.MatchProcessor

Framework and implementation modules
------------------------------------

.. autosummary::
   :toctree: generated

   PostManager

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
