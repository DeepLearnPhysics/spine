"""Visualization tools for SPINE data and results.

This module provides comprehensive visualization capabilities for neutrino physics data,
ML model outputs, and analysis results using Plotly for interactive 3D visualization.

**Core Visualization Functions:**
- `point`: General voxel and point cloud visualization (`scatter_points`)
- `particle`: Particle trajectory and property visualization
- `cluster`: Cluster analysis and display (`scatter_clusters`)
- `box`: Bounding box and region-of-interest visualization (`scatter_boxes`)
- `arrow`: Directional vector and momentum visualization

**Network & Graph Visualization:**
- `network`: Graph neural network topology visualization
  - `network_topology`: Display graph structure and connections
  - `network_schematic`: Simplified network diagrams

**Analysis & Output Visualization:**
- `out`: Model output and prediction visualization
- `evaluation`: Performance metrics and evaluation plots
- `train`: Training progress and loss visualization

**Geometry & Detector:**
- `geo`: Detector geometry and coordinate system visualization

**Layout & Styling:**
- `layout`: Pre-configured Plotly layouts (`plotly_layout3d`) for consistent styling

**Key Features:**
- Interactive 3D visualization with Plotly
- Automatic color coding and legends
- Support for large datasets with efficient rendering
- Detector geometry integration
- Truth vs. prediction comparison plots
- Export capabilities for publication-quality figures

**Example Usage:**
```python
from spine.vis import scatter_points, scatter_clusters, plotly_layout3d

# Visualize detector hits
fig = scatter_points(coordinates, features, color='energy')
fig.update_layout(plotly_layout3d)

# Visualize clustering results
fig = scatter_clusters(data, cluster_labels, method='dbscan')
```
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
