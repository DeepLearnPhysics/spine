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

Segmentation Metrics and Uncertainties
--------------------------------------

The ``segment_confusion`` and ``segment_distance`` report providers compute
voxel-weighted metrics from event confusion counts (predicted classes on rows,
true classes on columns). Both support ``classes`` and ``class_mapping``.
Per-class precision includes all truth classes in its predicted denominator;
recall includes all predicted classes in its truth denominator. Selecting a
class therefore does not hide its false positives or false negatives. Mapping
classes together treats predictions within the combined group as correct.

F1 is computed as ``2 * true_positive / (true_count + predicted_count)``.
An entirely absent class has null precision, recall and F1; a missed or
spuriously predicted class has zero F1. Macro F1 averages the defined class
F1 values, with the contributing names recorded in ``macro_f1_classes``.
Accuracy uses the selected true population. Counts are pooled before ratios
are computed, rather than averaging event metrics.

For ``segment_confusion``, the ``metrics`` JSON block contains these shared
metrics and their full-population denominators. Existing top-level accuracy,
precision and recall retain their conditional semantics for compatibility;
per-class ``f1`` is likewise conditional, while ``f1_overall``,
``precision_overall`` and ``predicted_overall`` expose full-population values.
Top-level ``macro_f1`` uses the shared full-population definition. The new
``<key>_metrics`` figure shows the shared per-class metrics.
For ``segment_distance``, each distance bin and the separate ``no_vertex``
summary contain the shared metrics directly. Distance plots show accuracy,
macro F1, precision, recall, F1 and voxel populations.

Enable optional percentile confidence intervals in either provider with:

.. code-block:: yaml

   metrics:
     segmentation:
       provider: segment_confusion
       source: "**/*segment_eval_summary.csv"
       bootstrap:
         replicates: 1000
         confidence: 0.95
         seed: 0
     vertex_distance:
       provider: segment_distance
       source: "**/*segment_eval_distance.csv"
       bootstrap:
         replicates: 1000
         confidence: 0.95
         seed: 0

Omit ``bootstrap`` to skip uncertainty estimation and retain streaming
reduction without storing event matrices. When enabled, the reducer retains
compact per-event confusion matrices in memory, scaling with the number of
events, bins and squared raw class count. The options shown above are defaults.
Replicates must be at least two, confidence must lie strictly between zero and
one, and seed must be a nonnegative integer.

Each replicate samples the original number of events with replacement, sums
their counts, then recomputes the metrics. All distance bins of an event share
the same sampling multiplicity, including its no-vertex record. CSV chunks do
not define sampling units. Event identities use ``file_index`` plus
``file_entry_index`` when available, then ``index`` with an optional
``file_index``, then ``run/subrun/event``. IDs are namespaced by CSV path, so
independent shards may reuse local IDs. Inputs should contain disjoint events
across shards; duplicated or correlated events across shards are not detected.
Missing or null event identities are rejected when bootstrapping is enabled.

Intervals appear in ``confidence_intervals`` beside their point estimates;
bootstrap metadata records the method, event count, seed and confidence level.
Each interval records ``lower``, ``upper``, ``valid_replicates`` and
``contributing_events``. Undefined replicate metrics are omitted from the
percentiles; macro F1 uses the defined classes in each replicate. Endpoints
are null for undefined estimates or fewer than two contributing events or
valid replicates. Sparse populations should therefore be interpreted alongside
these counts. Empty bins remain null rather than becoming zero-accuracy bins.
Distance plots show shaded pointwise intervals, not simultaneous confidence
bands; overall per-class plots show vertical intervals.

These intervals describe statistical sampling variation between independent
events. They do not include uncertainty from training or detector/model
systematics, and a degenerate bootstrap interval does not establish certainty.
