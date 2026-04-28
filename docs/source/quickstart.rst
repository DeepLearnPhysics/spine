Quick Start
===========

.. rst-class:: lead

This guide covers the standard way to run SPINE. For full reconstruction workflows, use the published SPINE container image. Run that image with Docker on local machines or with Apptainer / Singularity on HPC systems. Use an explicit release tag when you want a pinned runtime; when in doubt, use ``latest`` or omit the tag entirely. Local Python-only installs are more appropriate for post-processing, analysis, and visualization tasks.

Run SPINE In The Container
--------------------------

Pull the recommended runtime container with Docker:

.. code-block:: bash

   # Equivalent to: docker pull ghcr.io/deeplearnphysics/spine
   docker pull ghcr.io/deeplearnphysics/spine:latest

   # Pin to a specific release if needed
   docker pull ghcr.io/deeplearnphysics/spine:<release>

Or pull the same released image with Apptainer:

.. code-block:: bash

   # Equivalent to omitting the tag entirely in the Docker image reference
   apptainer pull spine_latest.sif docker://ghcr.io/deeplearnphysics/spine:latest

   # Or pin to a specific release
   apptainer pull spine_<release>.sif docker://ghcr.io/deeplearnphysics/spine:<release>

Run a configuration with Docker:

.. code-block:: bash

   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:latest \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

   # Or use a pinned release
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:<release> \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

Run the same configuration with Apptainer:

.. code-block:: bash

    # Latest published image
   apptainer exec --nv spine_latest.sif \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

   # Or use a pinned release
   apptainer exec --nv spine_<release>.sif \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

Run SPINE From Python
---------------------

If you want to inspect one entry interactively, start a shell in the container first:

.. code-block:: bash

   docker run --gpus all -it --rm -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:latest \
     bash

From that shell, you can open Python or Jupyter and use the same driver directly. A typical workflow is to process one entry and hand the result to :class:`spine.vis.out.Drawer` for visualization:

.. code-block:: python

   from spine.config import load_config_file
   from spine.driver import Driver
   from spine.vis.out import Drawer

   cfg = load_config_file("/workspace/config/train_uresnet.yaml")

   driver = Driver(cfg)
   data = driver.process(entry=0)

   drawer = Drawer(data, draw_mode="both")
   fig = drawer.get("particles")
   fig.show()

The same pattern works for other object families such as ``fragments`` or ``interactions``.

When To Use Local Python Installs
---------------------------------

If you only need to inspect outputs, make plots, or run downstream studies, a local install is often enough:

.. code-block:: bash

   pip install spine[viz]

For broader analysis or documentation work:

.. code-block:: bash

   pip install spine[all]

This mode is useful for ``spine.post``, ``spine.ana``, and ``spine.vis`` workflows, but it is not the recommended default for full reconstruction jobs when a released container image is available.

Next Steps
----------

- Review :doc:`installation` for the runtime options and tradeoffs
- Explore :doc:`config_loader` to understand SPINE configuration files
- Browse the API reference for the pipeline stage you are modifying or using
