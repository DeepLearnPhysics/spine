# Visualization Package

`spine.vis` provides visualization helpers used to build Plotly traces,
assemble domain-aware drawers, share plotting layouts, and render metric
figures such as confusion matrices and annotated heatmaps. It also provides a
renderer-neutral scene representation for high-volume 3D event displays.

The implementation lives in:

```text
src/spine/vis/trace/
src/spine/vis/drawer/
src/spine/vis/layout/
src/spine/vis/metric/
src/spine/vis/scene/
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

- `scene/`
  - Renderer-neutral, contiguous point-cloud layers
  - Object membership and numeric attributes are retained without requiring
    one render object per SPINE object
  - Configurable backends; Plotly is the built-in backend

## Design rule

Use the following boundary when deciding where new code belongs:

- Put code in `trace/` when it converts geometry-like inputs directly into
  Plotly traces.
- Put code in `drawer/` when it decides *what* to draw from SPINE objects or
  domain-aware structures, and then delegates to `trace/`.
- Put code in `scene/` when it describes *what is present in a 3D scene*
  independently of a plotting library, or converts that scene to a backend.

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

For large interactive point clouds, build a renderer-neutral scene first:

```python
scene = vis.Drawer(data, draw_mode="reco").get_scene(
    "particles",
    attr=["pid"],
    color_attr="pid",
)

# Existing notebook-friendly output
figure = scene.render("plotly")

# Optional legacy-style object splitting. The scene itself stays combined.
figure = scene.render("plotly", split_objects=True)
```

Prefer the combined form for large events. Plotly validates and copies every
nested trace independently, and its browser runtime maintains per-trace state.
Splitting hundreds of objects therefore adds substantial fixed overhead even
when the total point count is unchanged. Object boundaries remain available in
``PointLayer.object_ids`` and ``PointLayer.object_offsets`` so WebGL backends
can provide per-object picking and legends without creating separate buffers.

Browser applications can consume the neutral arrays without adding a browser
renderer to SPINE itself:

```python
layer = scene.views[0].layers[0]
positions = layer.positions
object_ids = layer.object_ids
attributes = layer.attributes
```

SPINE currently provides only the Python-side Plotly backend. Applications
such as Spinal Tap own their JavaScript renderer, transport and interactive
filter state. A reusable browser renderer can be extracted later if another
consumer, such as a Jupyter widget, needs the same implementation.

The first scene implementation covers the dominant object and raw point-cloud
path. Detector geometry and auxiliary optical/CRT glyphs continue to use the
established Plotly drawer while neutral line and mesh layers are developed.
For that reason, `Drawer.get` continues to use the complete legacy Plotly path
instead of delegating to `Drawer.get_scene`. Once the neutral model represents
detector geometry and every auxiliary layer, `get` can become the compatibility
façade for `get_scene(...).render("plotly")` without losing existing notebook
functionality.

Stored optical hypotheses can be overlaid with measured flashes in an output
event display:

```python
figure = vis.Drawer(data, draw_mode="reco", geo=geometry).get(
    "interactions",
    draw_flashes=True,
    draw_flash_hypotheses=True,
    optical_size_by_pe=False,
)
```

The default uses fixed detector sizes and distinct colorscales. Setting
`optical_size_by_pe=True` scales both measured and predicted optical responses
with one shared PE normalization.

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
