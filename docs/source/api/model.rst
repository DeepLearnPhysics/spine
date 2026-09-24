Model Module
============

The ``spine.model`` module contains the deep learning architectures used by SPINE for semantic segmentation, clustering, endpoint finding, graph construction, and end-to-end reconstruction tasks.

.. currentmodule:: spine.model

.. automodule:: spine.model
   :no-members:

Module Index
------------

The model package is built around configuration-driven instantiation through :class:`spine.model.ModelManager`. It includes convolutional, graph-based, and hybrid architectures tailored to sparse detector reconstruction rather than generic ML utilities.

.. autosummary::
   :toctree: generated

   ModelManager
   ValidationManager

Top-level model configurations
------------------------------

The ``model.name`` setting selects one of the registered network/loss pairs
below.  Their class pages expose the model-specific configuration dictionaries
and output contracts.

.. autosummary::
   :toctree: generated

   full_chain.FullChain
   full_chain.FullChainLoss
   graph_spice.GraphSPICE
   graph_spice.GraphSPICELoss
   grappa.GrapPA
   grappa.GrapPALoss
   image.ImageModel
   image.ImageLoss
   spice.SPICE
   spice.SPICELoss
   uresnet.UResNetSegmentation
   uresnet.SegmentationLoss
   uresnet.BayesianUResNet
   uresnet.BayesianSegmentationLoss

The :doc:`uresnet_ppn` page documents the registered ``uresnet_ppn`` network
and loss without importing them during the documentation build. The
:doc:`optional_runtime` page covers the full-chain segmentation provider and
AdaBound optimizers, which have the same runtime constraint.

.. toctree::
   :hidden:

   uresnet_ppn
   optional_runtime

Implementation modules
----------------------

.. autosummary::
   :toctree: generated

   manager
   validation
   checkpoint
   factories
   uresnet
   uresnet.bayes
   spice
   full_chain
   image
   grappa
   graph_spice
   cnn
   common
   sparse

The point-proposal implementation is available as
``spine.model.uresnet.ppn``. Its API is imported only in a model-capable
runtime because the package defines PyTorch type aliases and modules at import
time.

Multi-task loss balancing
-------------------------

Composite objectives may select ``sum`` (the backward-compatible default),
``fixed`` or learned ``uncertainty`` balancing through a sibling module block.
Loss producers declare objective families and batch activity internally; user
configuration contains only the policy and optional scientific priorities.

For standalone GrapPA, weight names match its prediction heads, such as
``node_type`` or ``edge``. Compound vertex prediction exposes distinct
``node_vertex_primary`` and ``node_vertex_reg`` objectives::

   model:
     name: grappa
     modules:
       grappa: ...
       grappa_loss: ...
       loss_balancing:
         name: uncertainty

For ``full_chain``, a top-level policy balances complete provider stages. The
weight names are the stage names from the normalized chain plan. These stage
objectives are marked as composite in code, so users do not assign a fictitious
classification or regression family to a mixed reconstruction stage::

   model:
     name: full_chain
     modules:
       chain: ...
       loss_balancing:
         name: fixed
         weights:
           segmentation: 1.0
           particle_aggregation: 0.5

A policy inside a native stage loss block instead balances that stage's leaf
objectives. Chain-wide and nested fixed priorities may be composed, but two
levels of learned uncertainty balancing are rejected because their scales are
not independently identifiable.

Gradient monitoring
-------------------

Training can collect post-backward gradient diagnostics without changing the
optimizer update. Tracking is disabled by default. When enabled, the global
group covers every trainable network and loss-module parameter. Optional named
groups select canonical parameter names with shell-style glob patterns::

   train:
     optimizer:
       name: Adam
       lr: 0.001
     gradient_tracking:
       interval: 100
       groups:
         backbone:
           - "network.uresnet.*"
         proposal:
           - "network.ppn.*"
         loss_balancer:
           - "loss.*log_variances.*"

Names begin with ``network.`` or ``loss.`` and are resolved after configured
parameter freezing but before distributed wrappers are installed. Each sampled
group reports its L2 norm, RMS value, maximum absolute value, missing-gradient
fraction and non-finite element count through the normal CSV and TensorBoard
backends. ``interval`` follows completed optimizer updates: an interval of 100
samples at zero-based iterations 99, 199 and so on. Rows between samples retain
the fixed CSV schema with ``gradient_sampled=0`` and ``NaN`` statistics.

Set ``include_global: false`` when only explicit groups are desired. Group names
must contain letters, numbers and underscores and must match at least one
trainable parameter, allowing misspelled or stale selections to fail during
model construction rather than silently producing empty diagnostics.

Gradient balancing with PCGrad
------------------------------

Training may apply `PCGrad <https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf>`_
after the model's scalar loss-balancing policy has been evaluated. PCGrad
computes one gradient per active objective and removes pairwise components with
a negative inner product before summing the task gradients. It therefore
addresses gradient direction conflicts rather than replacing ``sum``,
``fixed`` or ``uncertainty`` loss weighting. UResNet+PPN, standalone GrapPA
and ``full_chain`` currently publish the required objective metadata through a
shared internal interface::

   train:
     optimizer:
       name: Adam
       lr: 0.001
     gradient_balancing:
       name: pcgrad
       parameters: "network.*"
       seed: 0

The ``parameters`` option accepts one shell-style glob or a sequence of globs
using the same canonical ``network.`` names as gradient monitoring. It defaults
to all trainable network parameters. At each iteration, surgery is restricted
further to parameter tensors reached by at least two active objectives;
task-specific parameters and trainable loss-balancer variables retain their
ordinary combined-loss gradients. ``seed`` makes the paper's randomized task
order reproducible from the global iteration without adding checkpoint state.

PCGrad reports the active-task and shared-parameter-tensor counts, each task's
pre-projection norm within that shared subspace, mean pairwise cosine, conflict
fraction and projection count through the normal logging path. Passive gradient
monitoring may be enabled at the same time and observes the final gradients
after surgery. The additional per-task autograd traversals increase training
cost with the number of active objectives. Distributed training is rejected
explicitly for now because correct task-wise cross-rank reduction is not yet
implemented.

Sparse CNN lattice phase
------------------------

UResNet-based backbones can randomize their alignment with strided sparse
convolutions during training without moving detector coordinates or labels::

   uresnet:
     lattice:
       period: auto

One phase is sampled independently for each sparse image. The phase affects
only the backend coordinate map and is disabled automatically in evaluation
mode. Coordinate-convolution features, GraphSPICE and SPICE embeddings, and
published PPN or vertex coordinates remain in the input coordinate frame.
The automatic period follows the encoder depth and covers every alignment of
its deepest convolution lattice. An explicit integer or per-axis sequence may
be supplied instead when a study requires a fixed phase range.

GrapPA edge safety ceiling
--------------------------

GrapPA can remove the complete edge set of any graph entry which exceeds a
configured safety ceiling. The limit is applied after graph construction or
cache loading and before edge encoding, so it protects both live and
materialized execution:

.. code-block:: yaml

   model:
     modules:
       grappa:
         max_edge_count: 1600000

The boundary is inclusive: an entry with exactly ``max_edge_count`` edges is
retained, while an entry above it keeps its nodes and node objectives but has
all edges removed. GrapPA emits an original-axis ``edge_keep`` mask so cached
edge targets and validity masks remain aligned. The mask is composed with any
later training-only edge or node dropout. The former ``graph.max_count``
setting remains accepted for compatibility but is deprecated and routed
through this common path.

GrapPA graph augmentation
-------------------------

Graph augmentations are configured inside the GrapPA model and applied only
while the module is in training mode. Undirected edge dropout preserves the
adjacent reciprocal-edge convention used by SPINE graphs.

.. autosummary::
   :toctree: generated

   grappa.augment.EdgeDropout
   grappa.augment.EdgeSelection
