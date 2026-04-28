Data Module
===========

The ``spine.data`` module provides data structures and containers for handling sparse detector data, from low-level voxels to high-level physics objects.

Overview
--------

The data module is organized into several categories:

- **Output Data Structures**: High-level reconstructed and truth objects (fragments, particles, interactions)
- **Batch Data Structures**: Efficient batching utilities for ML training
- **Detector Data**: CRT, optical, trigger information
- **Metadata**: Run information, image metadata, and generic object lists

Output Data Structures
----------------------

These classes represent high-level physics objects produced by the reconstruction chain.
Each has both a reconstructed (Reco) and truth (Truth) variant with automatically merged
attributes from base classes.

Fragment Objects
~~~~~~~~~~~~~~~~

Fragments represent clusters of energy depositions that may correspond to intermediate objects
in the reconstruction hierarchy.

.. autoclass:: spine.data.out.RecoFragment
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spine.data.out.TruthFragment
   :members:
   :inherited-members:
   :show-inheritance:

Particle Objects
~~~~~~~~~~~~~~~~

Particles represent individual particles with full kinematic and identification information.

.. autoclass:: spine.data.out.RecoParticle
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spine.data.out.TruthParticle
   :members:
   :inherited-members:
   :show-inheritance:

Interaction Objects
~~~~~~~~~~~~~~~~~~~

Interactions represent complete neutrino interactions with all associated particles.

.. autoclass:: spine.data.out.RecoInteraction
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spine.data.out.TruthInteraction
   :members:
   :inherited-members:
   :show-inheritance:

Batch Data Structures
---------------------

Efficient batching utilities for machine learning training and inference.

.. autoclass:: spine.data.batch.TensorBatch
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spine.data.batch.EdgeIndexBatch
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: spine.data.batch.IndexBatch
   :members:
   :inherited-members:
   :show-inheritance:

Other Data Structures
---------------------

Additional data containers for detector information.

.. autoclass:: spine.data.Particle
   :members:
   :show-inheritance:

.. autoclass:: spine.data.Neutrino
   :members:
   :show-inheritance:

.. autoclass:: spine.data.CRTHit
   :members:
   :show-inheritance:

.. autoclass:: spine.data.Optical
   :members:
   :show-inheritance:

.. autoclass:: spine.data.Trigger
   :members:
   :show-inheritance:

.. autoclass:: spine.data.Meta
   :members:
   :show-inheritance:

.. autoclass:: spine.data.RunInfo
   :members:
   :show-inheritance:

.. autoclass:: spine.data.ObjectList
   :members:
   :show-inheritance:
