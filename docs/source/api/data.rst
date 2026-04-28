Data Module
===========

The ``spine.data`` module defines the shared data structures used across the SPINE pipeline, from detector-level input records to reconstructed and truth object hierarchies.

.. currentmodule:: spine.data

.. automodule:: spine.data
   :no-members:

Overview
--------

The canonical namespaces inside ``spine.data`` are:

- **LArCV data structures** under ``spine.data.larcv`` for detector records, generator truth, and metadata imported from LArCV-style sources
- **Output data structures** under ``spine.data.out`` for reconstructed and truth fragments, particles, and interactions
- **Batch data structures** under ``spine.data.batch`` for tensor, edge-index, and index batching in ML workflows
- **Generic list containers** under ``spine.data.list`` for lightweight container utilities

CRT, optical, trigger, run-information, and image-metadata classes are part of the LArCV-facing namespace in SPINE. They are detector or acquisition records, but they are not a separate top-level category outside ``spine.data.larcv``.

LArCV Data Structures
---------------------

These classes provide the low-level inputs and metadata that enter the reconstruction chain. They live canonically under ``spine.data.larcv`` and are re-exported from ``spine.data`` for convenience.

.. autosummary::
   :toctree: generated

   larcv.Particle
   larcv.Neutrino
   larcv.CRTHit
   larcv.Flash
   larcv.Trigger
   larcv.Meta
   larcv.RunInfo

Output Data Structures
----------------------

These classes represent the high-level object hierarchy produced by construction and refined by post-processing. Each object family has reconstructed and truth variants with shared metadata, units, and enum-aware fields inherited from common bases.

Fragment Objects
~~~~~~~~~~~~~~~~

Fragments represent clusters of energy depositions that may correspond to intermediate objects
in the reconstruction hierarchy.

.. autosummary::
   :toctree: generated

   out.RecoFragment
   out.TruthFragment

Particle Objects
~~~~~~~~~~~~~~~~

Particles represent individual particles with full kinematic and identification information.

.. autosummary::
   :toctree: generated

   out.RecoParticle
   out.TruthParticle

Interaction Objects
~~~~~~~~~~~~~~~~~~~

Interactions represent complete neutrino interactions with all associated particles.

.. autosummary::
   :toctree: generated

   out.RecoInteraction
   out.TruthInteraction

Batch Data Structures
---------------------

These containers support model-facing batching and unbatching for tensors, graph edges, and index collections.

.. autosummary::
   :toctree: generated

   batch.TensorBatch
   batch.EdgeIndexBatch
   batch.IndexBatch

Other Data Structures
---------------------

Additional generic containers defined directly under ``spine.data``.

.. autosummary::
   :toctree: generated

   ObjectList
