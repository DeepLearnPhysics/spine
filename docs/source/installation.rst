Installation
============

.. rst-class:: lead

SPINE has two practical installation modes:

- **Container-first** for full reconstruction, training, and inference workloads. This is the recommended way to run SPINE because the release-tagged SPINE image provides a compatible PyTorch ecosystem together with LArCV.
- **Local pip installation** for post-processing, analysis, visualization, documentation, or development workflows that do not need the full ML stack.

Recommended: Container Workflow
-------------------------------

For end-to-end SPINE usage, start from the SPINE image published for the matching SPINE release. Use an explicit release tag when you want a pinned runtime. When in doubt, use ``latest`` or omit the tag entirely, which is equivalent for Docker-style image references.

Docker path:

.. code-block:: bash

   # Equivalent to: docker pull ghcr.io/deeplearnphysics/spine
   docker pull ghcr.io/deeplearnphysics/spine:latest

   # Pin to a specific release when reproducibility matters
   docker pull ghcr.io/deeplearnphysics/spine:<release>

Apptainer / Singularity path:

.. code-block:: bash

   # Equivalent to omitting the tag entirely in the Docker image reference
   apptainer pull spine_latest.sif docker://ghcr.io/deeplearnphysics/spine:latest

   # Pin to a specific release when desired
   apptainer pull spine_<release>.sif docker://ghcr.io/deeplearnphysics/spine:<release>

Both paths use the same released SPINE image. Docker is usually simpler on workstations; Apptainer or Singularity is the usual choice on shared HPC systems.

This image includes the hard-to-maintain runtime stack used by SPINE, including PyTorch, CUDA support, torch-geometric packages, MinkowskiEngine, and LArCV2.

Run a SPINE job with Docker:

.. code-block:: bash

   # Latest published image
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:latest \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

   # Pinned release image
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:<release> \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

Run a SPINE job with Apptainer:

.. code-block:: bash

   # Latest published image
   apptainer exec --nv spine_latest.sif \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

   # Or run a pinned release image
   apptainer exec --nv spine_<release>.sif \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

Local pip Installation
----------------------

Use a local pip installation when you only need post-processing, analysis, visualization, documentation, or lightweight development without the full PyTorch and LArCV runtime.

Core Package
~~~~~~~~~~~~

For the minimal core package:

.. code-block:: bash

   pip install spine

This installs the base Python dependencies declared in ``pyproject.toml``. In practice, that includes the main numerical, tabular, configuration, and I/O packages used by SPINE, such as:

- ``numpy``
- ``scipy``
- ``pandas``
- ``PyYAML``
- ``h5py``
- ``numba``
- ``numexpr``
- ``psutil``
- ``tqdm``

Optional Python Extras
~~~~~~~~~~~~~~~~~~~~~~

Machine Learning Models
+++++++++++++++++++++++

Installing ML dependencies locally is possible, but it is not the preferred path for most users because the PyTorch ecosystem and LArCV compatibility are the hard part of getting SPINE running reliably.

If you do need to install model support outside the container:

.. code-block:: bash

   pip install spine[model]

You may still need to manage PyTorch, torch-geometric, torch-scatter, torch-cluster, MinkowskiEngine, and CUDA compatibility yourself, which is why the released SPINE image remains the recommended runtime.

Visualization Tools
+++++++++++++++++++

For analysis and visualization without the full ML stack:

.. code-block:: bash

   pip install spine[viz]

Development Tools
+++++++++++++++++

For package development and documentation work:

.. code-block:: bash

   pip install spine[dev]

All Dependencies
++++++++++++++++

To install all optional Python extras from PyPI:

.. code-block:: bash

   pip install spine[all]

This is useful for local post/ana/vis workflows, but it still does not replace the container as the recommended runtime for full SPINE jobs.

Python And Dependency Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Python 3.10 or newer
- The base dependencies installed by ``pip install spine``
- Optional extras depending on the workflow:

   - ``spine[viz]`` for plotting and visualization
   - ``spine[dev]`` for testing, formatting, packaging, and documentation
   - ``spine[all]`` for the union of the shipped Python extras

For GPU acceleration outside the container, you must ensure that CUDA-compatible PyTorch and the rest of the SPINE ML dependency stack are installed consistently.

Development From Source
~~~~~~~~~~~~~~~~~~~~~~~

For local development from source:

.. code-block:: bash

   git clone https://github.com/DeepLearnPhysics/spine.git
   cd spine
   pip install -e .[dev]

If you are developing reconstruction or model code, doing this inside the released SPINE container is usually the lowest-friction setup.
