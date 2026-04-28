SPINE Documentation
===================

Introduction
============

.. rst-class:: lead

`Scalable Particle Imaging with Neural Embeddings (SPINE) <https://github.com/DeepLearnPhysics/spine>`_ is a machine-learning reconstruction toolkit for particle imaging detectors, developed primarily for Liquid Argon Time Projection Chambers (LArTPCs). It combines configuration-driven I/O, deep neural network models, object construction, post-processing, analysis, and visualization into a single reconstruction workflow.

For full reconstruction, training, and inference workflows, SPINE is intended to run from the published SPINE container image released alongside each SPINE version. Use the release-tagged image ``ghcr.io/deeplearnphysics/spine:<release>`` when reproducibility matters. When in doubt, use ``ghcr.io/deeplearnphysics/spine:latest`` or omit the tag entirely, which is equivalent in Docker-style image references. Docker is the usual path on local machines; Apptainer or Singularity is the usual path on HPC systems. A local ``pip`` installation is most appropriate when you only need post-processing, analysis, visualization, or lightweight data inspection.

The package is organized around the :class:`spine.driver.Driver` pipeline:

- load detector inputs and labels
- run neural network inference or training
- unwrap batched outputs
- construct fragments, particles, and interactions
- apply post-processing and detector matching
- run analysis scripts and write results

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   Introduction <self>
   installation
   quickstart
   config_loader
   api/index

Getting Started
===============

The landing page should stay short and decision-oriented. The detailed setup and workflow instructions live in the dedicated guides linked below.

Installation
------------

For complete SPINE workflows, start from the released SPINE container image:

.. code-block:: bash

   # Equivalent to omitting the tag entirely
   docker pull ghcr.io/deeplearnphysics/spine:latest

   # Use an explicit release tag when you want a pinned runtime
   docker pull ghcr.io/deeplearnphysics/spine:<release>

On HPC systems, pull the same released image through Apptainer or Singularity:

.. code-block:: bash

   # Equivalent to omitting the tag entirely in the Docker image reference
   apptainer pull spine_latest.sif docker://ghcr.io/deeplearnphysics/spine:latest

   # Or pin to a specific release
   apptainer pull spine_<release>.sif docker://ghcr.io/deeplearnphysics/spine:<release>

For local ``pip`` installs, development workflows, and the full runtime discussion, see :doc:`installation`.

Quick Start
-----------

For the most direct end-to-end path, run SPINE from the released container with a configuration file:

.. code-block:: bash

   # Using the newest published image
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:latest \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

   # Or use a pinned release image
   docker run --gpus all -v $(pwd):/workspace \
     ghcr.io/deeplearnphysics/spine:<release> \
       spine --config /workspace/config/train_uresnet.yaml --source /workspace/data.h5

If you want to inspect one entry interactively, the Python entry point looks like this:

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

For the full interactive-container workflow, Apptainer examples, and the longer Python walkthrough, see :doc:`quickstart`.

SPINE also exposes lower-level modules for data structures, model components, construction, analysis, math helpers, and visualization, but the main user-facing workflow starts from the driver and configuration system.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
