"""Analysis script module.

This module handles scripts using the output of the reconstruction chain
and the post-processor module to produce analyses. This may include:
    - Reconstruction performance metrics 
    - Topology selection
    - Energy reconstruction quality
    - etc.
"""

from .manager import AnaManager
