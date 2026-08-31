Geometry Module
===============

The ``spine.geo`` package provides detector geometry interfaces used by reconstruction, post-processing, and visualization. It exposes the geometry data model together with the geometry manager that shares detector context across SPINE components.

.. automodule:: spine.geo
   :no-members:

Core Interfaces
---------------

.. autosummary::
   :toctree: generated
   :template: dataclass.rst

   base.Geometry

The geometry manager selects a detector configuration and exposes the shared
:class:`base.Geometry` instance to the rest of the pipeline.  Its page lists
the manager-level configuration parameters.

.. autosummary::
   :toctree: generated

   manager.GeoManager

Detector components
-------------------

Geometry YAML is nested: ``Geometry.tpc`` configures
:class:`detector.TPCDetector`, while the optional ``optical`` and ``crt``
blocks configure :class:`detector.OptDetector` and
:class:`detector.CRTDetector`.  The structured pages below separate stored
fields, computed properties, and inherited box geometry, including the
descriptions that are lost in a conventional dataclass attribute table.

.. autosummary::
   :toctree: generated
   :template: dataclass.rst

   detector.Plane
   detector.Box
   detector.tpc.TPCChamber
   detector.tpc.TPCModule
   detector.TPCDetector
   detector.optical.OpticalVolume
   detector.OptDetector
   detector.crt.CRTPlane
   detector.CRTDetector

Utilities
---------

.. autosummary::
   :toctree: generated

   factories
   utils
