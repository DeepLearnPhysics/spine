"""Post-processing module.

This module handles post-processors building upon the ouptut of the ML-based
reconstruction chain to produce high-level quantities. This may include:
    - Particle direction reconstruction
    - Range-based energy estimation
    - Optical flash matching
    - Cosmic ray tagger matching
    - etc.
"""

from .manager import PostManager
