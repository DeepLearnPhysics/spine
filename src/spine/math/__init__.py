"""Fast, Numba-accelerated math routines.

This includes multiple submodules:
- `base.py` includes basic functions, as found in numpy or scipy.special
- `linalg.py` includes linear algebra routines, as found in numpy.linalg
- `distance.py` includes distance functions, as found in scipy.distance
- `graph.py` includes graph routines, as found in scipy.csgraph
- `cluster.py` includes cluster functions, as found in sklearn.cluster
- `metrics` includes statistical and clustering evaluation metrics
- `match.py` includes overlap-based assignment primitives
"""

# Expose submodules
from . import cluster, decomposition, distance, graph, linalg, match, metrics, neighbors

# Expose all base functions directly
from .base import *
