"""Visualization helpers, trace builders, and domain-aware drawers.

The top-level :mod:`spine.vis` namespace re-exports the public visualization
API while the implementation is organized internally into:

- :mod:`spine.vis.trace` for low-level Plotly trace builders
- :mod:`spine.vis.drawer` for higher-level object and detector drawers
- :mod:`spine.vis.layout` for shared Plotly and Matplotlib styling
- :mod:`spine.vis.metric` for metric-specific plotting helpers
- :mod:`spine.vis.scene` for renderer-neutral 3D scenes and backends
"""

from importlib import import_module as _import_module

from .drawer import *
from .layout import *
from .metric import *
from .scene import *
from .trace import *

# Wildcard re-exports can expose implementation modules with the same names as
# these public packages. Bind the package attributes explicitly so dotted API
# paths always resolve to their documented modules.
drawer = _import_module(".drawer", __name__)
layout = _import_module(".layout", __name__)
metric = _import_module(".metric", __name__)
scene = _import_module(".scene", __name__)
trace = _import_module(".trace", __name__)

del _import_module
