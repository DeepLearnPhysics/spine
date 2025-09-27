"""Module which holds all detector component geometries.

This includes:
- :class:`TPCDetector` for a set of organized TPCs
- :class:`OptDetector` for a set of organized light collection detectors
- :class:`CRTDetector` for a set of organized CRT planes
"""

from .crt import CRTDetector
from .optical import OptDetector
from .tpc import TPCDetector
