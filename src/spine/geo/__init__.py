"""Detector geometry tools.

This package provides geometry containers and helpers for detector-aware
coordinate queries. Use :class:`Geometry` for standalone geometry objects and
:class:`GeoManager` when application code needs a shared process-wide geometry
instance.
"""

from .base import Geometry
from .manager import GeoManager
