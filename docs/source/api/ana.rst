Analysis Module
===============

The ``spine.ana`` module contains the analysis layer that runs after reconstruction, construction, and post-processing. It is intended for configuration-driven studies of reconstruction quality, detector performance, and physics-facing event content.

.. currentmodule:: spine.ana

.. automodule:: spine.ana
   :no-members:

Module Index
------------

Use this package when the goal is to turn reconstructed SPINE outputs into metrics, derived tables, plots, or physics-study inputs.

Event and columnar processing
-----------------------------

Analysis scripts implement ``AnaBase.process`` for the default event-oriented
path. Scripts which can preserve their semantics on projected product chunks
may additionally implement ``AnaBase.process_columnar``. The
``AnaBase.run_columnar`` wrapper applies the same required/optional key
contract without slicing individual events, while
``AnaManager(columnar=True)`` checks every configured module during
initialization.

Columnar support is deliberately opt-in. A manager refuses columnar execution
if any configured script only implements the event path, preventing partially
written analysis outputs. Enable the mode on the HDF5 reader with
``columnar: true``; this is an all-or-nothing execution policy for the analysis
configuration. Individual scripts declare their required fields through
``columnar_requests`` and receive projected columns plus event-boundary
metadata through ``process_columnar``.

Configurable analysis scripts
-----------------------------

Each class below can be selected by its ``name`` attribute under ``ana:``.
Its reference page documents all constructor options and defaults.

.. autosummary::
   :toctree: generated

   calib.MCSCalibAna
   diag.GraphEdgeLengthAna
   diag.PointCompletenessAna
   diag.ShowerStartDEdxAna
   diag.TrackCompletenessAna
   metric.ClusterAna
   metric.FlashMatchingAna
   metric.PointProposalAna
   metric.SegmentAna
   script.SaveAna

Framework and implementation modules
------------------------------------

.. autosummary::
   :toctree: generated

   AnaManager

.. autosummary::
   :toctree: generated

   base
   manager
   template
   factories
   calib
   diag
   metric
