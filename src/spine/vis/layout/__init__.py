"""Shared Plotly layout, color, and styling helpers for visualization code.

The :mod:`spine.vis.layout` package exposes a compact public API while keeping
the implementation split by concern:

- :mod:`spine.vis.layout.plotly` for 3D scene and subplot layout helpers
- :mod:`spine.vis.layout.matplotlib` for Matplotlib and seaborn styling
- :mod:`spine.vis.layout.colors` for shared palettes and color formatting
"""

from .colors import *
from .matplotlib import *
from .plotly import *
