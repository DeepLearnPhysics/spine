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
