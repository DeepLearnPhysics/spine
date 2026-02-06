Installation
============

SPINE can be installed via pip from PyPI. The package is organized with optional dependencies to keep the core installation minimal.

Core Installation
-----------------

For basic functionality with sparse data handling and custom mathematical operations:

.. code-block:: bash

   pip install spine

Optional Dependencies
---------------------

Machine Learning Models
~~~~~~~~~~~~~~~~~~~~~~~

For neural network models and training capabilities:

.. code-block:: bash

   pip install spine[model]

This includes PyTorch, PyTorch Geometric, and other ML dependencies.

Visualization Tools
~~~~~~~~~~~~~~~~~~~

For plotting and visualization capabilities:

.. code-block:: bash

   pip install spine[viz]

This includes Plotly, Matplotlib, and related packages.

Development Tools
~~~~~~~~~~~~~~~~~

For package development:

.. code-block:: bash

   pip install spine[dev]

This includes Sphinx, pytest, and development utilities.

All Dependencies
~~~~~~~~~~~~~~~~

To install everything:

.. code-block:: bash

   pip install spine[all]

System Requirements
-------------------

- Python 3.8+
- NumPy
- SciPy
- Numba (for performance-critical computations)

For GPU acceleration with neural networks, ensure CUDA-compatible PyTorch is installed.

Development Installation
------------------------

For local development from source:

.. code-block:: bash

   git clone https://github.com/DeepLearnPhysics/spine.git
   cd spine
   pip install -e .[dev]
