"""Module with fast, Numba-accelerated, compiles math routines.

This includes multiple submodules:
- `base.py` includes basic functions, as found in numpy or scipy.special
- `linalg.py` includes linear algebra routines, as found in numpy.linalg
- `distance.py` includes distance functions, as found in scipy.distance
- `graph.py` includes graph routines, as found in scipy.csgraph
- `cluster.py` includes cluster functions, as found in skleran.cluster
"""

# Expose all base functions directly
from .base import *

# Expose submodules
from . import cluster
from . import decomposition
from . import distance
from . import graph
from . import linalg
from . import neighbors
