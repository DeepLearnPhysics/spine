.. image:: _static/img/spine-logo-dark.png
   :alt: SPINE Logo
   :width: 400px
   :align: center

SPINE Documentation
===================

**Scalable Particle Imaging with Neural Embeddings (SPINE)** is a Python package for machine learning in particle physics, providing tools for 3D reconstruction in sparse detectors like Liquid Argon Time Projection Chambers (LArTPCs).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Installation
============

Install SPINE from PyPI:

.. code-block:: bash

   # Core functionality
   pip install spine-ml

   # With neural network modeling
   pip install spine-ml[model]

   # With visualization tools
   pip install spine-ml[viz]

   # For development
   pip install spine-ml[dev]

   # Everything
   pip install spine-ml[all]

Quick Start
===========

.. code-block:: python

   import spine
   import numpy as np

   # Use SPINE's custom math functions
   from spine.math.cluster import DBSCAN
   from spine.math.decomposition import PCA

   # Example: Cluster 3D points
   points = np.random.rand(100, 3).astype(np.float32)

   # Fast numba-accelerated DBSCAN
   dbscan = DBSCAN(eps=0.1, min_samples=5)
   labels = dbscan.fit_predict(points)

   # Principal component analysis
   pca = PCA(n_components=2)
   components, variance = pca.fit(points)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
