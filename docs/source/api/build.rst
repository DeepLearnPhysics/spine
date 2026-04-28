Construct Module
================

The ``spine.construct`` module provides tools for building higher-level analysis objects from reconstruction primitives.

.. note::
   This module was formerly named ``build`` and configuration files still use ``build:`` for backward compatibility.

Overview
--------

The construct module converts raw ML model outputs into structured physics objects:

- **BuildManager**: Central factory coordinating all builders
- **FragmentBuilder**: Creates Fragment objects from clustering results
- **ParticleBuilder**: Creates Particle objects with full kinematics
- **InteractionBuilder**: Creates complete Interaction hierarchies

Build Manager
-------------

Central coordination for object construction.

.. autoclass:: spine.construct.BuildManager
   :members:
   :show-inheritance:

Builders
--------

Individual builders for each object type. All builders inherit from ``BuilderBase``
and share common configuration options.

Fragment Builder
~~~~~~~~~~~~~~~~

.. autoclass:: spine.construct.fragment.FragmentBuilder
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: build_reco, build_truth, load_reco, load_truth

Particle Builder
~~~~~~~~~~~~~~~~

.. autoclass:: spine.construct.particle.ParticleBuilder
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: build_reco, build_truth, load_reco, load_truth

Interaction Builder
~~~~~~~~~~~~~~~~~~~

.. autoclass:: spine.construct.interaction.InteractionBuilder
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: build_reco, build_truth, load_reco, load_truth
