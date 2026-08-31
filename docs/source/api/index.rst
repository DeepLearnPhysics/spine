API Reference
=============

This section follows the public package layout introduced for SPINE 1.0. The
driver orchestrates I/O, model execution, object construction,
post-processing, analysis, logging, and output writing. Shared packages expose
the data model, detector geometry, clustering, physics, numerical kernels, and
the deliberately small set of cross-layer utilities.

Configuration-oriented entries are indexed by concrete class, rather than
stopping at a package page.  Follow the class selected by a YAML ``name`` (or
the manager for a top-level block) to see its accepted parameters, defaults,
attributes, and methods.  In particular, :doc:`post`, :doc:`ana`,
:doc:`calib`, :doc:`io`, :doc:`model`, and :doc:`geo` enumerate the
implementations that can surface through the main configuration.

The focused package pages and top-level entry points are the supported
user-facing reference. :doc:`modules` is intentionally exhaustive for
development and debugging; inclusion there alone does not imply API stability.
See :doc:`../support` for the compatibility boundary.

.. toctree::
   :maxdepth: 2

   spine
   config
   constants
   data
   build
   geo
   io
   logging
   ana
   calib
   cluster
   math
   model
   physics
   post
   utils
   vis
   modules
