"""Post-processing tools for ML-based neutrino reconstruction.

This module transforms raw ML model outputs into high-level physics quantities,
serving as the bridge between deep learning predictions and physics analysis.

Core post-processing management:

- ``PostManager`` coordinates configurable reconstruction cleanup pipelines.

Major processing categories:

- Particle reconstruction, including direction, energy, PID, and vertex refinement.
- Detector matching for optical flashes, CRT hits, and trigger information.
- Event-level classification, containment studies, and quality metrics.

Output products:

- Reconstructed particles and interactions with refined kinematics.
- Truth matching associations and quality flags.
- Detector-matching metadata and calibration-aware quantities.

Example configuration
---------------------

.. code-block:: yaml

   post:
     chain:
       - name: ParticleBuilder
         track_min_length: 3.0
       - name: EnergyReconstructor
         method: csda
       - name: FlashMatcher
         time_window: 10.0

This module is essential for converting raw ML outputs into analysis-ready
physics objects with proper uncertainties and quality assessments.
"""

from .manager import PostManager
