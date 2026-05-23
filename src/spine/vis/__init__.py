"""Visualization helpers, trace builders, and domain-aware drawers.

The top-level :mod:`spine.vis` namespace re-exports the public visualization
API while the implementation is organized internally into:

- :mod:`spine.vis.trace` for low-level Plotly trace builders
- :mod:`spine.vis.drawer` for higher-level object and detector drawers
- :mod:`spine.vis.layout` for shared Plotly and Matplotlib styling
- :mod:`spine.vis.metric` for metric-specific plotting helpers
"""

from .drawer import *
from .layout import *
from .metric import *
from .trace import *
