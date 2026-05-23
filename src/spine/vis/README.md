# Visualization Package

`spine.vis` provides visualization helpers used to build Plotly traces,
assemble domain-aware drawers, share plotting layouts, and render metric
figures such as confusion matrices and annotated heatmaps.

The implementation lives in:

```text
src/spine/vis/trace/
src/spine/vis/drawer/
src/spine/vis/layout/
src/spine/vis/metric/
```

The top-level `spine.vis` namespace re-exports the intended public API, while
the subpackages keep the implementation split by responsibility.

## Package layout

- `trace/`
  - Low-level Plotly trace builders
  - Inputs are typically arrays, coordinates, cluster indices, colors, and
    hover labels
  - Examples: point clouds, boxes, ellipsoids, cones, hulls

- `drawer/`
  - Higher-level visualization entrypoints
  - Inputs are typically SPINE domain objects, detector geometry, graph
    structures, or training logs
  - These modules compose lower-level trace helpers into complete views

- `layout/`
  - Shared Plotly and Matplotlib styling/layout helpers
  - Scene configuration, subplot layouts, palettes, and plotting style

- `metric/`
  - Metric-specific plotting helpers
  - Confusion matrices, heatmaps, and related annotation helpers

## Design rule

Use the following boundary when deciding where new code belongs:

- Put code in `trace/` when it converts geometry-like inputs directly into
  Plotly traces.
- Put code in `drawer/` when it decides *what* to draw from SPINE objects or
  domain-aware structures, and then delegates to `trace/`.

Examples:

- `trace/point.py`: draw points
- `trace/box.py`: draw boxes from bounds
- `drawer/geo.py`: draw detector geometry from a `Geometry` object
- `drawer/lite.py`: draw lite particles/interactions from SPINE objects

## Public API

The top-level `spine.vis` namespace re-exports the public visualization API.
Subpackages also re-export their own intended public symbols through their
`__init__.py` files.

Typical usage patterns:

```python
import spine.vis as vis

traces = vis.scatter_points_3d(points, color=values)
drawer = vis.GeoDrawer()
```

or, when you want to import from the implementation layer directly:

```python
from spine.vis.trace.point import scatter_points_3d
from spine.vis.drawer.geo import GeoDrawer
```

## Typing conventions

Several visualization inputs allow either one shared scalar value or a
per-element sequence. Shared aliases and validators for those patterns live in
`trace/utils.py`.

Examples include:

- `ColorInput`
- `HoverTextInput`
- `IntensityInput`

When adding new visualization helpers, prefer those shared aliases and helper
functions over ad hoc `np.isscalar(...)` checks.

## Tests

The test tree mirrors the package structure:

- `test/test_vis/test_trace/`
- `test/test_vis/test_drawer/`
- `test/test_vis/test_layout/`
- `test/test_vis/test_metric/`

When adding a new module under `src/spine/vis`, add or update the
corresponding tests in the mirrored subtree.
