"""Visualization tools for SPINE data and results.

This module provides comprehensive visualization capabilities for neutrino physics data,
ML model outputs, and analysis results using Plotly for interactive 3D visualization.

Core visualization modules:

- ``point`` for voxel and point-cloud displays.
- ``particle`` and ``cluster`` for physics-object rendering.
- ``box``, ``arrow``, ``cone``, and related primitives for geometric overlays.
- ``out`` for reconstruction output inspection.
- ``geo`` for detector geometry context.

Key features:

- Interactive 3D Plotly-based visualization.
- Shared layouts and color handling.
- Detector-aware overlays and truth-versus-reco comparisons.

Example
-------

.. code-block:: python

   from spine.vis import scatter_points, scatter_clusters, layout3d

   fig = scatter_points(coordinates, features, color="energy")
   fig.update_layout(layout3d)

   fig = scatter_clusters(data, cluster_labels, method="dbscan")
"""

from .arrow import *
from .box import *
from .cluster import *
from .evaluation import *
from .geo import *
from .layout import *
from .network import *
from .out import *
from .particle import *
from .point import *
from .train import *
