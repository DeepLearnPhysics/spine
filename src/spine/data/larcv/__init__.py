"""LArCV input data structures.

This module contains data structures that represent information parsed from
LArCV files, including detector-level information (CRT hits, optical flashes),
particle-level information, metadata, and run information.
"""

from .crt import *
from .meta import *
from .neutrino import *
from .optical import *
from .particle import *
from .run_info import *
from .trigger import *

__all__ = [
    # CRT
    "CRTHit",
    # Meta
    "Meta",
    # Neutrino
    "Neutrino",
    # Optical
    "Flash",
    # Particle
    "Particle",
    # Run Info
    "RunInfo",
    # Trigger
    "Trigger",
]
