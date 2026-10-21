Visualization Module
====================

The ``spine.vis`` module provides interactive visualization tools for detector inputs, reconstructed objects, truth labels, geometry, and training or evaluation products. Output drawers can return either Plotly figures or renderer-neutral scenes, allowing notebook users to retain Plotly while browser applications upload compact typed geometry directly to another renderer.

.. currentmodule:: spine.vis

.. automodule:: spine.vis
   :no-members:

Module Index
------------

Use this package to inspect detector point clouds, overlay reconstructed or truth objects, and visualize model or evaluation outputs in notebooks and analysis workflows.

.. autosummary::
   :toctree: generated

   drawer
   layout
   metric
   scene
   trace

Figure Layout
-------------

Pass layout defaults to :class:`Drawer`, then override them for individual
:meth:`Drawer.get` calls. Each call returns an independent figure; overrides
do not change the drawer defaults or earlier figures.

.. code-block:: python

   drawer = Drawer(data, draw_mode="both", height=800, title="Event display")
   fig = drawer.get(
       "particles",
       title="Selected event",
       titles=["Reconstruction", "Truth"],
       margin={"l": 20},
   )
   untitled = drawer.get("particles", title=None)

``title`` labels the entire figure and accepts a string or a Plotly title
dictionary. ``Drawer.get(titles=...)`` labels the two split panels. In the
lower-level ``layout3d`` helper, ``titles`` instead labels the three axes;
constructor ``Drawer(..., titles=...)`` options are forwarded to that helper.

Single and split views default to ``width=None``, ``autosize=True``, and
``height=800``. The automatic width fits the containing element, while a
concrete height keeps 3D views usable in containers without a defined height.
Explicit widths, heights, and ``autosize`` settings are preserved. Notebook or
web embedding must also allow Plotly to resize when its container changes.

Nonempty figure titles reserve 60 pixels at the top. Split-panel titles reserve
another 42 pixels. Explicit margin edges win, including ``margin={"t": 0}``;
unspecified edges retain automatic defaults. A per-call margin dictionary
replaces the constructor margin dictionary rather than merging with it.
Other per-call options likewise replace constructor options by keyword.

``dual_figure3d`` copies supplied layouts and preserves their dimensions and
margin edges unless explicitly overridden. Omitting its width or height keeps
the supplied value; passing ``None`` clears that dimension. Without a supplied
layout, it uses the same responsive width and 800-pixel height defaults.

Renderer-Neutral Scenes
-----------------------

Use :meth:`Drawer.get_scene` when the consumer, rather than SPINE, owns the
renderer. A scene contains one or two named views and typed point, marker,
line, vector, indexed-mesh, or box layers. Point layers retain compact object
offsets and numeric recoloring attributes. The registered ``plotly`` backend
can convert the same scene back into a figure:

.. code-block:: python

   drawer = Drawer(data, draw_mode="both")
   scene = drawer.get_scene(
       "particles",
       attr=["pid", "shape"],
       draw_end_points=True,
       draw_directions=True,
   )
   fig = scene.render("plotly")
   fig.show()
