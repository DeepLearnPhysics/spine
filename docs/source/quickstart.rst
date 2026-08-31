Quick Start
===========

.. rst-class:: lead

This guide covers the standard ways to run SPINE. Install the Python package
locally for data inspection, post-processing, analysis, and visualization. For
full reconstruction and training workflows, use the published SPINE container
image with Docker on local machines or Apptainer / Singularity on HPC systems.
Use an explicit release tag when you want a pinned runtime; when in doubt, use
``latest`` or omit the tag entirely.

.. _inspect-existing-output:

Inspect Existing SPINE Output
-----------------------------

Install the core Python package from PyPI:

.. code-block:: bash

   python -m pip install spine

To run the visualization example below, include the ``viz`` extra:

.. code-block:: bash

   python -m pip install "spine[viz]"

Suppose ``spine_output.h5`` contains reconstructed particles and the supporting
``points`` and ``depositions`` products. Save the following as ``inspect.yaml``:

.. code-block:: yaml

   base:
     iterations: -1

   io:
     reader:
       name: hdf5
       file_keys: /path/to/spine_output.h5
       keep_open: false

   build:
     mode: reco
     units: cm
     fragments: false
     particles: true
     interactions: false

The driver can then load and prepare any selected entry for visualization:

.. code-block:: python

   from spine.config import load_config_file
   from spine.driver import Driver
   from spine.vis import Drawer

   cfg = load_config_file("inspect.yaml")
   data = Driver(cfg).process(entry=0)

   drawer = Drawer(data, draw_mode="reco")
   fig = drawer.get("particles")
   fig.show()

The HDF5 reader reconstructs the serialized particle records. The ``build``
block additionally reconnects each particle to its long-form point,
deposition, and index arrays, which are required by consumers such as
:class:`spine.vis.Drawer`. Enable ``interactions`` or ``fragments`` when those
object families are present and needed; interaction construction also requires
``particles: true``.

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
       spine --config /workspace/config/full_chain/full_chain_regression.yaml \
       --source /workspace/input.root

   # Or use a pinned release
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:<release> \
       spine --config /workspace/config/full_chain/full_chain_regression.yaml \
       --source /workspace/input.root

On Apple Silicon macOS systems, add ``--platform=linux/amd64`` to ``docker
run`` when using the published SPINE image:

.. code-block:: bash

    docker run --platform=linux/amd64 --gpus all -v $(pwd):/workspace \
       ghcr.io/deeplearnphysics/spine:<release> \
          spine --config /workspace/config/full_chain/full_chain_regression.yaml \
          --source /workspace/input.root

For Jupyter notebook/lab use specifically, avoid the Docker Desktop
combination of Apple Virtualization Framework **with** Rosetta enabled. Apple
Virtualization Framework without Rosetta and Docker VMM have both been
verified to work for Jupyter with the published image, while normal SPINE CLI
commands continue to work with either setting.

Run the same configuration with Apptainer:

.. code-block:: bash

    # Latest published image
   apptainer exec --nv spine_latest.sif \
       spine --config /workspace/config/full_chain/full_chain_regression.yaml \
       --source /workspace/input.root

   # Or use a pinned release
   apptainer exec --nv spine_<release>.sif \
       spine --config /workspace/config/full_chain/full_chain_regression.yaml \
       --source /workspace/input.root

Run SPINE From Python
---------------------

If you want to inspect one entry interactively, start a shell in the container first:

.. code-block:: bash

   docker run --gpus all -it --rm -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:latest \
     bash

   # On Apple Silicon macOS, add --platform=linux/amd64 before the image name.

From that shell, you can open Python or Jupyter and use the reader-mode driver
shown in :ref:`inspect-existing-output` directly. Random-access
``process(entry=...)`` calls are intended for reader and inference workflows;
training configurations consume their configured loader sequentially instead.

The same pattern works for other object families such as ``fragments`` or ``interactions``.
Applications that own their 3D renderer can call
``drawer.get_scene("particles")`` instead; the returned typed scene can still
be rendered in a notebook with ``scene.render("plotly")``.

Run JupyterLab Or Classic Notebook
----------------------------------

To launch JupyterLab directly from the published container:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/workspace \
       ghcr.io/deeplearnphysics/spine:<release> \
       jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

On Apple Silicon macOS systems, add ``--platform=linux/amd64`` before the
image name:

.. code-block:: bash

    docker run --platform=linux/amd64 --gpus all -it --rm -p 8888:8888 -v $(pwd):/workspace \
       ghcr.io/deeplearnphysics/spine:<release> \
       jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

If you specifically want the classic notebook UI instead of Lab:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/workspace \
       ghcr.io/deeplearnphysics/spine:<release> \
       jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

Open the URL printed by Jupyter in your browser on the host machine.

When To Use Local Python Installs
---------------------------------

If you only need to inspect outputs, make plots, or run downstream studies, a local install is often enough:

.. code-block:: bash

   pip install spine[viz]

For broader analysis or documentation work:

.. code-block:: bash

   pip install spine[dev]

This mode is useful for ``spine.post``, ``spine.ana``, and ``spine.vis`` workflows, but it is not the recommended default for full reconstruction jobs when a released container image is available.

Next Steps
----------

- Review :doc:`installation` for the runtime options and tradeoffs
- Follow :doc:`workflows` for training, inference, full reconstruction, and resume
- Use :doc:`operations` before qualifying or scaling a production campaign
- Read :doc:`pipeline` for the ownership and execution boundaries
- Read :doc:`data_model` before consuming reconstructed and truth objects
- Use :doc:`configuration` to map each YAML block to its accepted parameters
- Explore :doc:`config_loader` for includes, overrides, and path handling
- Start with :doc:`troubleshooting` when a production run fails
- Read :doc:`support` for API stability and compatibility boundaries
- Browse the API reference for the pipeline stage you are modifying or using
