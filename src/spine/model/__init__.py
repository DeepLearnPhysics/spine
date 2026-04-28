"""Machine learning models for neutrino physics reconstruction.

This module handles the construction, training, and execution of deep learning models
for liquid argon time projection chamber (LArTPC) data analysis.

Model management:

- ``ModelManager`` coordinates model instantiation, training, evaluation, and checkpoints.

Supported model families:

- ``UResNet`` for semantic segmentation.
- ``PPN`` for endpoint proposals.
- ``SPICE`` for point-cloud instance clustering.
- ``GrapPA`` and related graph models for relational reconstruction.
- End-to-end chain models for full reconstruction workflows.

Key features:

- Modular configuration-driven model assembly.
- Support for sparse and dense convolutions.
- Graph neural network components.
- Mixed precision and distributed training support.

Example
-------

.. code-block:: python

   from spine.model import ModelManager

   manager = ModelManager(config)
   manager.train(train_loader, val_loader)
   results = manager.forward(data_batch)

The module integrates with the broader SPINE ecosystem for data I/O,
visualization, and post-processing workflows.
"""

from .manager import ModelManager
