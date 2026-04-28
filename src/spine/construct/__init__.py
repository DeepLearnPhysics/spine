"""Data structure construction and analysis object building.

This module handles the construction of high-level analysis data classes from
ML reconstruction outputs and provides I/O capabilities for analysis workflows.

Note:

- Configuration files still use the ``build:`` key for backward compatibility, while the Python module is named ``construct``.

Core construction management:

- ``BuildManager`` coordinates object construction, truth matching, and validation.

Constructed object hierarchy:

- ``RecoFragment`` and ``TruthFragment`` for fragment-level objects.
- ``RecoParticle`` and ``TruthParticle`` for particle-level objects.
- ``RecoInteraction`` and ``TruthInteraction`` for event-level hierarchies.

Example
-------

.. code-block:: python

   from spine.construct import BuildManager

   builder = BuildManager(config)
   objects = builder.build_from_ml_chain(ml_results)

   interaction = objects.interactions[0]
   particles = interaction.particles
   fragments = particles[0].fragments

This module provides the foundation for all downstream physics analysis
by creating well-structured, validated analysis objects with full provenance.
"""

from .manager import BuildManager
