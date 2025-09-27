"""Module with fast, Numba-accelerated, compiles math routines.

This includes multiple submodules:
- `base.py` includes basic functions, as found in numpy or scipy.special
- `linalg.py` includes linear algebra routines, as found in numpy.linalg
- `distance.py` includes distance functions, as found in scipy.distance
- `graph.py` includes graph routines, as found in scipy.csgraph
- `cluster.py` includes cluster functions, as found in skleran.cluster
- `metrics.py` includes clustering evaluation metrics
"""

# Expose submodules
from . import cluster, decomposition, distance, graph, linalg, metrics, neighbors

# Expose all base functions directly
from .base import *
