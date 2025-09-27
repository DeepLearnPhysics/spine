Quick Start
===========

This guide covers the basic usage of SPINE's core functionality.

Basic Math Operations
---------------------

SPINE provides high-performance mathematical operations optimized with Numba:

Clustering with DBSCAN
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from spine.math.cluster import DBSCAN

   # Generate sample 3D point cloud
   points = np.random.rand(1000, 3).astype(np.float32)

   # Apply DBSCAN clustering
   dbscan = DBSCAN(eps=0.1, min_samples=5)
   labels = dbscan.fit_predict(points)

   print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

Principal Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spine.math.decomposition import PCA

   # Perform PCA on data
   pca = PCA(n_components=2)
   components, explained_variance = pca.fit(points)

   # Transform data
   transformed = pca.transform(points)

Distance Metrics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from spine.math import metrics

   # Compute pairwise distances
   distances = metrics.pairwise_distances(points[:100], points[:50])

   # Various distance metrics available
   euclidean_dist = metrics.euclidean_distance(points[0], points[1])

Graph Construction
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spine.math.neighbors import kneighbors_graph

   # Build k-nearest neighbors graph
   graph = kneighbors_graph(points, n_neighbors=5, mode='distance')

Working with Sparse Data
------------------------

SPINE is designed for sparse 3D data common in particle physics:

.. code-block:: python

   # Typical sparse data format: [x, y, z, features...]
   sparse_data = np.array([
       [0, 0, 0, 1.5],  # voxel at (0,0,0) with value 1.5
       [1, 0, 0, 2.3],  # voxel at (1,0,0) with value 2.3
       [0, 1, 0, 1.8],  # etc...
   ])

   # Extract coordinates and values
   coords = sparse_data[:, :3]
   values = sparse_data[:, 3]

Performance Tips
----------------

- Use float32 for coordinates when possible (faster with Numba)
- SPINE's algorithms are optimized for sparse data patterns
- GPU acceleration available for neural network components (with [model] installation)

Next Steps
----------

- Explore the full API documentation for advanced features
- See examples for machine learning workflows
- Check out visualization tools with the [viz] installation
