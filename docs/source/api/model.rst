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
