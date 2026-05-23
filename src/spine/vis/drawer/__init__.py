"""High-level visualization entrypoints.

This package groups drawers that understand SPINE domain objects, detector
geometry, graph semantics, or training logs and compose lower-level trace
helpers into full figures.
"""

from .geo import *
from .lite import *
from .network import *
from .out import *
from .particle import *
from .train import *
