"""Machine learning models for neutrino physics reconstruction.

This module handles the construction, training, and execution of deep learning models
for liquid argon time projection chamber (LArTPC) data analysis.

Model management:

- ``ModelManager`` coordinates model instantiation, training, evaluation, and checkpoints.

Supported model families:

- ``UResNet`` for semantic segmentation.
- ``PPN`` for endpoint proposals.
- ``SPICE`` for spatial-embedding instance clustering.
- ``GraphSPICE`` for point-cloud instance clustering.
- ``GrapPA`` and related graph models for relational reconstruction.
- Whole-image classification models.
- End-to-end chain models for full reconstruction workflows.

Key features:

- Modular configuration-driven model assembly.
- Support for sparse and dense convolutions.
- Graph neural network components.
- Distributed training support.

Example
-------

.. code-block:: python

   from spine.driver import Driver

   driver = Driver(config)
   results = driver.process(iteration=0)

The module integrates with the broader SPINE ecosystem for data I/O,
visualization, and post-processing workflows.
"""

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManifest,
    checkpoint_sha256,
    inspect_checkpoint,
    promote_checkpoint,
    verify_checkpoint,
)
from .export import export_model_weights
from .manager import ModelManager
from .validation import ValidationManager

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointManifest",
    "ModelManager",
    "ValidationManager",
    "checkpoint_sha256",
    "inspect_checkpoint",
    "promote_checkpoint",
    "verify_checkpoint",
    "export_model_weights",
]
