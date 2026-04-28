Construct Module
================

The ``spine.construct`` module converts model outputs and intermediate reconstruction products into the high-level object hierarchy used throughout SPINE analysis. Although configuration files still use the ``build:`` key for backward compatibility, the Python package name is ``spine.construct``.

.. currentmodule:: spine.construct

.. automodule:: spine.construct
   :no-members:

.. note::
   This module was formerly named ``build`` and configuration files still use ``build:`` for backward compatibility.

Overview
--------

Construction is where sparse predictions become physics objects with relationships and provenance:

- **BuildManager** coordinates object construction, truth matching, and validation
- **FragmentBuilder** groups low-level outputs into fragment objects
- **ParticleBuilder** assembles fragment-level information into particle candidates
- **InteractionBuilder** builds event-level interaction hierarchies

This stage sits between model execution and post-processing, and defines the reconstructed and truth object model consumed by downstream analysis tools.

Reference
---------

.. autosummary::
   :toctree: generated

   BuildManager
   fragment.FragmentBuilder
   particle.ParticleBuilder
   interaction.InteractionBuilder
