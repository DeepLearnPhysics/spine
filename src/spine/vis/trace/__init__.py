"""Trace-building helpers for visualization primitives.

This package contains low-level routines that convert geometry, point clouds,
and clustered point sets into Plotly traces. These helpers are routinely
consumed directly and also serve as building blocks for higher-level drawers.
"""

from .arrow import *
from .box import *
from .cluster import *
from .cone import *
from .cylinder import *
from .ellipsoid import *
from .hull import *
from .point import *
from .utils import *
