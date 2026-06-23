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

   manager
   factories
   uresnet
   uresnet_ppn
   spice
   full_chain
   image
   singlep
   vertex
   grappa
   graph_spice
   bayes_uresnet
   layer
