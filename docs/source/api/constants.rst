Constants Module
================

The ``spine.constants`` package is the canonical home for stable SPINE conventions that should not depend on detector configuration, run conditions, or campaign-specific choices. It collects tensor column layouts, enums, label maps, physical constants, and sentinel values.

.. automodule:: spine.constants
   :no-members:

Submodules
----------

Use focused submodules when you want one category of constants without importing the full convenience namespace.

.. autosummary::
   :toctree: generated

   columns
   enums
   factory
   labels
   physics
   sentinels
